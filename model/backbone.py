# Transformer Backbone: GroupEmbedding + Transformer
import torch
import torch.nn as nn
import math


class GroupEmbedding(nn.Module):
    def __init__(self, groups, d_model):
        super().__init__()
        self.groups = groups
        self.encoders = nn.ModuleList([
            nn.Linear(len(g), d_model) for g in groups
        ])

    def forward(self, x, masked_group_idx=None):
        # x: (batch, num_features)
        tokens = []
        for i, (g, enc) in enumerate(zip(self.groups, self.encoders)):
            feat = x[:, g]  # (batch, group_size)
            token = enc(feat)  # (batch, d_model)
            tokens.append(token)
        out = torch.stack(tokens, dim=1)  # (batch, num_groups, d_model)
        if masked_group_idx is not None:
            # 把被选中的 group token 置零 (batch,) indices
            batch_idx = torch.arange(out.size(0), device=out.device)
            out[batch_idx, masked_group_idx] = 0.0
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBackbone(nn.Module):
    def __init__(self, groups, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = GroupEmbedding(groups, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=len(groups))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model
        self.num_groups = len(groups)

    def forward(self, x, masked_group_idx=None):
        tokens = self.embedding(x, masked_group_idx)
        tokens = self.pos_enc(tokens)
        return self.transformer(tokens)


class GroupPretrainHead(nn.Module):
    def __init__(self, groups, d_model):
        super().__init__()
        self.groups = groups
        self.decoders = nn.ModuleList([
            nn.Linear(d_model, len(g)) for g in groups
        ])
        self.max_group_size = max(len(g) for g in groups)

    def forward(self, hidden, chosen_group_idx):
        # hidden: (batch, num_groups, d_model)
        # chosen_group_idx: (batch,)
        batch_size = hidden.size(0)
        device = hidden.device

        preds = torch.zeros(batch_size, self.max_group_size, device=device)
        valid = torch.zeros(batch_size, self.max_group_size, dtype=torch.bool, device=device)

        # 按组分批处理，避免 Python 循环过慢
        for gi in range(len(self.groups)):
            mask = (chosen_group_idx == gi)
            if not mask.any():
                continue
            group_hidden = hidden[mask, gi, :]  # (n_selected, d_model)
            group_pred = self.decoders[gi](group_hidden)  # (n_selected, group_size)
            gs = len(self.groups[gi])
            preds[mask, :gs] = group_pred
            valid[mask, :gs] = True

        return preds, valid
