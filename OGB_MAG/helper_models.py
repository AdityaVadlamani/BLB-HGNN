import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *


class MaskedFNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MaskedFNN, self).__init__()
        self.weight = nn.Parameter(
            kaiming_uniform_(torch.randn(in_dim, out_dim), a=math.sqrt(5)),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.randn(out_dim) * 1e-5, requires_grad=True)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_norm = nn.LayerNorm(normalized_shape=out_dim)

    def forward(self, x):
        new_shape = list(x.shape)
        new_shape[-1] = self.out_dim
        res = torch.mm(x.view(-1, self.in_dim), self.weight).view(-1, self.out_dim)
        mask = (res.sum(dim=-1) != 0).view(-1, 1).repeat(1, self.out_dim)
        res += torch.where(mask, self.bias, 0)
        return F.relu(self.layer_norm(res.view(new_shape)))


class BipartiteEmbedder(nn.Module):
    def __init__(
        self,
        feat_dim,
        neighbor_feat_dim,
        embed_dim,
        num_transformer_heads,
        num_transformer_layers,
        transformer_dropout,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.node_fnn = MaskedFNN(feat_dim, embed_dim)
        self.neighbor_fnn = MaskedFNN(neighbor_feat_dim, embed_dim)

        self.transformer = Transformer(
            d_model=embed_dim,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
            dropout=transformer_dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        neighbors_features: torch.Tensor,
        neighbors_weights: torch.Tensor,
    ) -> torch.Tensor:
        transformer_input = torch.concat(
            (
                self.node_fnn(node_features.unsqueeze(1)),
                self.neighbor_fnn(neighbors_features) * neighbors_weights,
            ),
            dim=1,
        )
        return self.transformer(transformer_input)[:, 0, :]


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.25, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.t_pe = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x.add_(self.pe[:, : x.size(1)])  # type: ignore
        return self.t_pe(self.dropout(x))


class Transformer(nn.Module):
    def __init__(
        self, d_model, num_heads, num_layers, dropout, use_pre_layer_norm=True
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
            norm_first=use_pre_layer_norm,
            dropout=dropout,
        )

        self.pos_encoder = PositionalEncoder(d_model=d_model)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=(not use_pre_layer_norm),
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Re-initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, calculate_gain("relu"))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        global_token = generate_global_token(
            embeddings.size(0), embeddings.size(2), embeddings.device
        )
        src = torch.concat((global_token, embeddings), dim=1)
        src_key_padding_mask = torch.count_nonzero(src, dim=2) == 0
        src = self.pos_encoder(src)
        return self.encoder(src=src, src_key_padding_mask=src_key_padding_mask)


def generate_global_token(batch_size, embed_dim, device):
    global_token = torch.empty((batch_size, 1, embed_dim), device=device)
    normal_(
        global_token,
        mean=0.0,
        std=embed_dim**-0.5,
    )
    return global_token
