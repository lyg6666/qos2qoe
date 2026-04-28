from model.backbone import TransformerBackbone, GroupPretrainHead
from model.dcn import DCN
import torch
import torch.nn as nn


class PretrainModel(nn.Module):
    def __init__(self, groups, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.backbone = TransformerBackbone(groups, d_model, n_heads, n_layers, dropout)
        self.pretrain_head = GroupPretrainHead(groups, d_model)

    def forward(self, x, chosen_group_idx):
        hidden = self.backbone(x, chosen_group_idx)
        preds, valid = self.pretrain_head(hidden, chosen_group_idx)
        return preds, valid


class FullModel(nn.Module):
    def __init__(self, groups, d_model, n_heads, n_layers,
                 cross_layers=3, deep_dims=None, dropout=0.1,
                 seq_len=1, output_dim=1):
        super().__init__()
        self.backbone = TransformerBackbone(groups, d_model, n_heads, n_layers, dropout)
        self.d_model = d_model
        self.seq_len = seq_len
        num_groups = len(groups)

        flat_dim = num_groups * d_model

        if seq_len > 1:
            self.proj = nn.Linear(flat_dim, 256)
            self.temporal = nn.LSTM(input_size=256, hidden_size=256,
                                    num_layers=1, batch_first=True)
            dcn_input_dim = 256
        else:
            self.proj = None
            self.temporal = None
            dcn_input_dim = flat_dim

        self.dcn = DCN(dcn_input_dim, cross_layers, deep_dims, dropout, output_dim)

    def forward(self, x):
        if self.seq_len > 1:
            B, T, F = x.shape
            x_flat = x.view(B * T, F)
            hidden = self.backbone(x_flat)  # (B*T, num_groups, d_model)
            hidden = hidden.view(B * T, -1)  # (B*T, flat_dim)
            hidden = self.proj(hidden)       # (B*T, 256)
            hidden = hidden.view(B, T, -1)   # (B, T, 256)
            _, (h_n, _) = self.temporal(hidden)
            out = h_n[-1]                    # (B, 256)
        else:
            hidden = self.backbone(x)        # (batch, num_groups, d_model)
            out = hidden.view(hidden.size(0), -1)  # (batch, flat_dim)

        return self.dcn(out)
