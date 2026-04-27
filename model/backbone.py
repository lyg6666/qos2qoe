# Transformer Backbone: FeatureEmbedding + Transformer
import torch
import torch.nn as nn
import math


class FeatureEmbedding(nn.Module):
	def __init__(self, num_features, d_model):
		super().__init__()
		self.projections = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
		self.mask_token = nn.Parameter(torch.randn(d_model))
		self.num_features = num_features
		self.d_model = d_model

	def forward(self, x, mask_positions=None):
		# x: (batch, num_features)
		# mask_positions: list[list[int]] 每个样本要 mask 的特征索引列表，或 None
		tokens = []
		for i in range(self.num_features):
			tokens.append(self.projections[i](x[:, i:i+1]))
		tokens = torch.stack(tokens, dim=1)  # (batch, num_features, d_model)

		if mask_positions is not None:
			for b, positions in enumerate(mask_positions):
				for pos in positions:
					tokens[b, pos] = self.mask_token
		return tokens


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
	def __init__(self, num_features, d_model, n_heads, n_layers, dropout=0.1):
		super().__init__()
		self.embedding = FeatureEmbedding(num_features, d_model)
		self.pos_enc = PositionalEncoding(d_model, max_len=num_features)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
			dropout=dropout, batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
		self.d_model = d_model
		self.num_features = num_features

	def forward(self, x, mask_positions=None):
		tokens = self.embedding(x, mask_positions)
		tokens = self.pos_enc(tokens)
		output = self.transformer(tokens)
		return output


class PretrainHead(nn.Module):
	# 组级掩码预测头：从所有 mask 位置的 token 预测原始值
	def __init__(self, d_model):
		super().__init__()
		self.head = nn.Linear(d_model, 1)

	def forward(self, hidden_states, mask_positions):
		# hidden_states: (batch, num_features, d_model)
		# mask_positions: list[list[int]]
		# 返回: list[Tensor]，每个 Tensor 形状为 (group_size,)
		preds = []
		for b, positions in enumerate(mask_positions):
			masked_hidden = hidden_states[b, positions]  # (group_size, d_model)
			pred = self.head(masked_hidden).squeeze(-1)  # (group_size,)
			preds.append(pred)
		return preds
