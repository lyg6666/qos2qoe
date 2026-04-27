from model.backbone import TransformerBackbone, PretrainHead
from model.dcn import DCN
import torch
import torch.nn as nn


class PretrainModel(nn.Module):
	def __init__(self, num_features, d_model, n_heads, n_layers, dropout=0.1):
		super().__init__()
		self.backbone = TransformerBackbone(num_features, d_model, n_heads, n_layers, dropout)
		self.pretrain_head = PretrainHead(d_model)

	def forward(self, x, chosen_padded, chosen_valid):
		hidden = self.backbone(x)
		preds = self.pretrain_head(hidden, chosen_padded, chosen_valid)
		return preds


class FullModel(nn.Module):
	def __init__(self, num_features, d_model, n_heads, n_layers,
				 cross_layers=3, deep_dims=None, dropout=0.1,
				 seq_len=1, output_dim=1):
		super().__init__()
		self.backbone = TransformerBackbone(num_features, d_model, n_heads, n_layers, dropout)
		self.num_features = num_features
		self.d_model = d_model
		self.seq_len = seq_len

		flat_dim = num_features * d_model  # 4352

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
			# x: (batch, seq_len, num_features)
			B, T, F = x.shape
			x_flat = x.view(B * T, F)
			hidden = self.backbone(x_flat)  # (B*T, num_features, d_model)
			hidden = hidden.view(B * T, -1)  # (B*T, flat_dim)
			hidden = self.proj(hidden)  # (B*T, 256)
			hidden = hidden.view(B, T, -1)  # (B, T, 256)
			_, (h_n, _) = self.temporal(hidden)
			out = h_n[-1]  # (B, 256)
		else:
			# x: (batch, num_features)
			hidden = self.backbone(x)  # (batch, num_features, d_model)
			out = hidden.view(hidden.size(0), -1)  # (batch, flat_dim)

		return self.dcn(out)
