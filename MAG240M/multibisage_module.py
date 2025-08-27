from typing import Any, List

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper_models import BipartiteEmbedder, Transformer
from lightning_utilities.core.rank_zero import rank_zero_info
from losses import FocalLoss
from torchmetrics.classification import Accuracy


class MultiBiSageModule(L.LightningModule):
    def __init__(
        self,
        /,
        *,
        name,
        num_classes,
        raw_feat_dim,
        raw_neigh_feat_dims,
        m2v_feat_dim,
        embedding_dim,
        hidden_layer_dim,
        num_transformer_heads,
        num_transformer_layers,
        dropout,
        lr=1e-4,
        weight_decay=1e-1,
        gamma=1.0,
        job_id="most_recent",
        train_label_weights=None,
    ) -> None:
        super().__init__()
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.job_id = job_id

        self.register_buffer("train_label_weights", train_label_weights)

        # Sub-models and parameters
        self.bipartite_models = nn.ModuleList(
            [
                BipartiteEmbedder(
                    raw_feat_dim,
                    raw_neigh_dim,
                    m2v_feat_dim,
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

        # Accuracy metrics
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
        embeddings = []
        for idx, bip_embedder in enumerate(self.bipartite_models):
            embeddings.append(
                bip_embedder(
                    nodes_features,
                    neighbors_features[idx],
                    neighbors_weights[idx],
                    m2v_features,
                    m2v_neighbors_features[idx],
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

        self.hparams["optim.lr"] = self.lr
        self.hparams["optim.weight_decay"] = self.weight_decay

        return [optimizer]

    def on_fit_start(self):
        rank_zero_info(f"MultiBiSage params:\n{self.hparams}")

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
        features, labels, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats = batch
        outputs = self.forward(
            features, neigh_feats, neigh_weights, m2v_feats, m2v_neigh_feats
        )
        outputs = self.classifier(outputs)
        outputs = torch.softmax(outputs, -1)
        self.test_acc.update(outputs, labels)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.test_labelwise_acc.update(outputs, labels)

        self.test_results_dict["y_pred"].extend(torch.argmax(outputs, dim=-1).tolist())
        self.test_results_dict["y_true"].extend(labels.tolist())

    def on_test_epoch_end(self) -> None:
        self.logger.log_table(
            "labelwise_test_accs",
            columns=[f"label_{i}_acc" for i in range(self.hparams.num_classes)],
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
