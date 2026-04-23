# Transformer Backbone: FeatureEmbedding + Transformer
import torch
import torch.nn as nn
import math


class FeatureEmbedding(nn.Module):
	# 每个特征一个独立Linear(1, d_model)，输出68个token
	def __init__(self, num_features, d_model):
		super().__init__()
		self.projections = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
		self.mask_token = nn.Parameter(torch.randn(d_model))  # 可学习的mask token
		self.num_features = num_features
		self.d_model = d_model

	def forward(self, x, mask_indices=None):
		# x: (batch, num_features)  mask_indices: (batch,)
		tokens = []
		for i in range(self.num_features):
			tokens.append(self.projections[i](x[:, i:i+1]))  # (batch, d_model)
		tokens = torch.stack(tokens, dim=1)  # (batch, num_features, d_model)
		# 将mask位置替换为mask_token
		if mask_indices is not None:
			batch_idx = torch.arange(x.shape[0], device=x.device)
			tokens[batch_idx, mask_indices] = self.mask_token
		return tokens


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=128):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

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

	def forward(self, x, mask_indices=None):
		# x: (batch, num_features)  -> (batch, num_features, d_model)
		tokens = self.embedding(x, mask_indices)
		tokens = self.pos_enc(tokens)
		output = self.transformer(tokens)  # (batch, num_features, d_model)
		return output


class PretrainHead(nn.Module):
	# 预训练头：从mask位置的token预测原始值
	def __init__(self, d_model):
		super().__init__()
		self.head = nn.Linear(d_model, 1)

	def forward(self, hidden_states, mask_indices):
		# hidden_states: (batch, num_features, d_model)
		batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
		masked_hidden = hidden_states[batch_idx, mask_indices]  # (batch, d_model)
		return self.head(masked_hidden).squeeze(-1)  # (batch,)
