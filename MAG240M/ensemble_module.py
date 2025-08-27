from copy import deepcopy
from typing import Any, List, Literal, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from losses import FocalLoss
from multibisage_module import MultiBiSageModule
from torchmetrics.classification import Accuracy
from tqdm import tqdm


class EnsembleModule(L.LightningModule):
    def __init__(
        self,
        /,
        *,
        name,
        num_classes: int,
        models: List[MultiBiSageModule],
        ensemble_method: Union[
            Literal["avg_params"],
            Literal["avg_params_finetune"],
            Literal["voting"],
            Literal["deep_ensemble"],
        ],
        weights: Optional[List[float]] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-1,
        gamma: float = 1.0,
        job_id="most_recent_averaged",
        train_label_weights=None,
    ) -> None:
        super().__init__()
        self.name = name
        self.ensemble_method = ensemble_method
        self.models = models
        self.lr = lr
        self.weight_decay = weight_decay
        self.job_id = job_id
        self.num_classes = num_classes

        self.register_buffer("train_label_weights", train_label_weights)

        if ensemble_method in ["voting", "deep_ensemble"]:
            self.classifier = [model.classifier for model in models]
        else:
            with torch.no_grad():
                state_dicts = [m.state_dict() for m in models]
                if weights is None:
                    weights = [1.0 / len(models)] * len(models)
                averaged = {}
                for key in state_dicts[0]:
                    averaged[key] = torch.sum(
                        torch.stack(
                            [
                                sd[key] * weight
                                for sd, weight in zip(state_dicts, weights)
                            ]
                        ),
                        axis=0,
                    )
                self.avg_model = deepcopy(models[0])
                self.avg_model.load_state_dict(averaged)
                self.classifier = self.avg_model.classifier

        # Evaluation metrics
        self.train_acc = Accuracy(
            num_classes=num_classes, task="multiclass", average="micro"
        )
        self.val_acc = Accuracy(
            num_classes=num_classes, task="multiclass", average="micro"
        )
        self.test_acc = Accuracy(
            num_classes=num_classes, task="multiclass", average="micro"
        )

        # Metric to see labelwise accuracy
        self.test_labelwise_acc = Accuracy(
            num_classes=num_classes, task="multiclass", average=None
        )

        # Final result evaluator
        self.test_results_dict = {
            "y_pred": [],
            "y_true": [],
        }

        self.train_loss_fn = FocalLoss(alpha=train_label_weights, gamma=gamma)

        self.hparams["train_loss_fn"] = self.train_loss_fn.__class__.__name__
        self.hparams["train_loss_fn.gamma"] = gamma

    def forward(
        self,
        nodes_features: torch.Tensor,
        neighbors_features: List[torch.Tensor],
        neighbors_weights: List[torch.Tensor],
        m2v_features: torch.Tensor,
        m2v_neighbors_features: List[torch.Tensor],
    ) -> torch.Tensor:
        if self.ensemble_method in ["voting", "deep_ensemble"]:
            embs = [
                m(
                    nodes_features,
                    neighbors_features,
                    neighbors_weights,
                    m2v_features,
                    m2v_neighbors_features,
                )
                for m in self.models
            ]
            return embs

        return self.avg_model(
            nodes_features,
            neighbors_features,
            neighbors_weights,
            m2v_features,
            m2v_neighbors_features,
        )

    def configure_optimizers(self) -> Any:
        without_decay = []
        with_decay = []
        for p in self.parameters():
            if p.requires_grad:
                if p.dim() > 1:
                    with_decay.append(p)
                else:
                    without_decay.append(p)

        optimizer = optim.AdamW(
            [{"params": with_decay}, {"params": without_decay, "weight_decay": 0}],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.hparams["ensemble.optim.defaults"] = vars(optimizer)["defaults"]
        return [optimizer]

    def training_step(self, batch, batch_idx):
        features, labels, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats = batch
        outputs = self.forward(
            features, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats
        )
        outputs = self.classifier(outputs)

        loss = self.train_loss_fn(outputs, labels)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        outputs = torch.softmax(outputs, -1)

        self.train_acc.update(outputs, labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats = batch
        outputs = self.forward(
            features, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats
        )
        outputs = self.classifier(outputs)

        val_loss = F.cross_entropy(outputs, labels)
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        outputs = torch.softmax(outputs, -1)
        self.val_acc.update(outputs, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        if self.ensemble_method in ["voting", "deep_ensemble"]:
            features, labels = batch[0].to(self.device), batch[1].to(self.device)
            neigh_feats, neigh_weights = (
                [y.to(self.device) for y in x] for x in batch[2:4]
            )
            m2v_feats = batch[4].to(self.device)
            m2v_neigh_feats = [y.to(self.device) for y in batch[5]]
        else:
            features, labels, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats = (
                batch
            )

        outputs = self.forward(
            features, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats
        )
        if self.ensemble_method == "voting":
            outputs = (
                torch.stack(
                    [
                        clf(outputs[i]).argmax(dim=-1)
                        for i, clf in enumerate(self.classifier)
                    ]
                )
                .mode(dim=0)
                .values
            )
            outputs = outputs.to(self.test_acc.device)
            labels = labels.to(self.test_acc.device)
            self.test_acc.update(outputs, labels)
        elif self.ensemble_method == "deep_ensemble":
            outputs = torch.mean(
                torch.stack([clf(outputs[i]) for i, clf in enumerate(self.classifier)]),
                dim=0,
            )
            outputs = torch.softmax(outputs, -1)
            self.test_acc.update(outputs, labels)
        else:
            outputs = self.classifier(outputs)
            outputs = torch.softmax(outputs, -1)
            self.test_acc.update(outputs, labels)
            self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

            self.test_labelwise_acc.update(outputs, labels)

            self.test_results_dict["y_pred"].extend(outputs.argmax(dim=-1).tolist())
            self.test_results_dict["y_true"].extend(labels.tolist())

    def on_test_epoch_end(self):
        if "avg_params" not in self.ensemble_method:
            return self.test_acc.compute()
        else:
            self.logger.log_table(
                "labelwise_test_accs",
                columns=[f"label_{i}_acc" for i in range(self.num_classes)],
                data=[self.test_labelwise_acc.compute()],
            )

            self.logger.log_table(
                "test_predictions",
                columns=["y_pred", "y_true"],
                data=[
                    list(x)
                    for x in zip(
                        self.test_results_dict["y_pred"],
                        self.test_results_dict["y_true"],
                    )
                ],
            )

    # Function to perform test step without a Trainer object
    def inference_test(self, test_dataloader):
        for batch_idx, batch in tqdm(enumerate(test_dataloader)):
            self.test_step(batch=batch, batch_idx=batch_idx)
        return self.on_test_epoch_end()
