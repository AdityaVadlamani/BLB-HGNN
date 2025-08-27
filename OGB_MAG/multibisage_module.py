from typing import Any, List

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper_models import BipartiteEmbedder, Transformer
from lightning_utilities.core.rank_zero import rank_zero_info
from losses import FocalLoss
from torchmetrics.classification import F1Score


class MultiBiSageModule(L.LightningModule):
    def __init__(
        self,
        /,
        *,
        name,
        num_classes,
        raw_feat_dim,
        raw_neigh_feat_dims,
        embedding_dim,
        hidden_layer_dim,
        num_transformer_heads,
        num_transformer_layers,
        dropout,
        is_multilabel=False,
        lr=1e-4,
        weight_decay=1e-1,
        use_lr_scheduler=False,
        gamma=1.0,
        train_label_weights=None,
    ) -> None:
        super().__init__()
        self.name = name
        self.is_multilabel = is_multilabel
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.use_lr_scheduler = use_lr_scheduler
        self.bipartite_models = nn.ModuleList(
            [
                BipartiteEmbedder(
                    raw_feat_dim,
                    raw_neigh_dim,
                    embedding_dim,
                    num_transformer_heads,
                    num_transformer_layers,
                    dropout,
                )
                for raw_neigh_dim in raw_neigh_feat_dims
            ]
        )
        self.encoder = Transformer(
            d_model=embedding_dim,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_layer_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(num_features=hidden_layer_dim),
            nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(num_features=hidden_layer_dim),
            nn.Linear(in_features=hidden_layer_dim, out_features=num_classes),
        )

        if is_multilabel:
            self.train_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.train_loss_fn = FocalLoss(alpha=train_label_weights, gamma=gamma)

        # Metrics
        task = "multilabel" if self.is_multilabel else "multiclass"
        self.train_f1 = F1Score(
            num_classes=num_classes, num_labels=num_classes, task=task, average="micro"
        )
        self.val_f1 = F1Score(
            num_classes=num_classes, num_labels=num_classes, task=task, average="micro"
        )
        self.test_f1 = F1Score(
            num_classes=num_classes, num_labels=num_classes, task=task, average="micro"
        )

        # Metric to see labelwise accuracy
        self.test_labelwise_f1 = F1Score(
            num_classes=num_classes, num_labels=num_classes, task=task, average=None
        )

        # Final test results
        self.test_results_dict = {
            "y_pred": [],
            "y_true": [],
        }

    def forward(
        self,
        nodes_features: torch.Tensor,
        neighbors_features: List[torch.Tensor],
        neighbors_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        embeddings = []
        for idx, bip_embedder in enumerate(self.bipartite_models):
            embeddings.append(
                bip_embedder(
                    nodes_features, neighbors_features[idx], neighbors_weights[idx]
                )
            )

        encoder_input = torch.stack(embeddings, dim=1)
        return F.normalize(self.encoder(encoder_input)[:, 0, :], dim=1)

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
        self.hparams["optim.defaults"] = vars(optimizer)["defaults"]

        if self.use_lr_scheduler:
            steps_per_epoch = int(
                len(self.trainer.datamodule.train_dataloader())
                / self.trainer.accumulate_grad_batches
            )

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=steps_per_epoch * self.trainer.max_epochs,
                eta_min=self.lr / 10,
            )

            return [optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

        return [optimizer]

    def on_fit_start(self):
        rank_zero_info(f"MultiBiSage params:\n{self.hparams}")

    def training_step(self, batch, batch_idx):
        features, labels, neigh_feats, neigh_weights = batch
        outputs = self.forward(features, neigh_feats, neigh_weights)
        outputs = self.classifier(outputs)

        if self.is_multilabel:
            labels = labels.float()

        loss = self.train_loss_fn(outputs, labels)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.is_multilabel:
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.softmax(outputs, -1)

        self.train_f1.update(outputs, labels)
        self.log(
            "train_f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, neigh_feats, neigh_weights = batch
        outputs = self.forward(features, neigh_feats, neigh_weights)
        outputs = self.classifier(outputs)

        if self.is_multilabel:
            labels = labels.float()
            val_loss = self.train_loss_fn(outputs, labels)
        else:
            val_loss = F.cross_entropy(outputs, labels)

        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.is_multilabel:
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.softmax(outputs, -1)

        self.val_f1.update(outputs, labels)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        features, labels, neigh_feats, neigh_weights = batch
        outputs = self.forward(features, neigh_feats, neigh_weights)
        outputs = self.classifier(outputs)

        if self.is_multilabel:
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.softmax(outputs, -1)

        self.test_f1.update(outputs, labels)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

        self.test_labelwise_f1.update(outputs, labels)

        self.test_results_dict["y_pred"].extend(torch.argmax(outputs, dim=-1).tolist())
        self.test_results_dict["y_true"].extend(labels.tolist())

    def on_test_epoch_end(self) -> None:
        self.logger.log_table(
            "labelwise_test_f1s",
            columns=[f"label_{i}_f1" for i in range(self.hparams.num_classes)],
            data=[self.test_labelwise_f1.compute()],
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