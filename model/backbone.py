# Transformer Backbone: FeatureEmbedding + Transformer
import torch
import torch.nn as nn
import math


class FeatureEmbedding(nn.Module):
	def __init__(self, num_features, d_model):
		super().__init__()
		self.weight = nn.Parameter(torch.empty(num_features, d_model))
		self.bias = nn.Parameter(torch.zeros(num_features, d_model))
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		self.num_features = num_features
		self.d_model = d_model

	def forward(self, x):
		# x: (batch, num_features)
		# 一次广播矩阵乘替代 68 次独立 Linear 调用
		return x.unsqueeze(-1) * self.weight + self.bias  # (batch, num_features, d_model)


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

	def forward(self, x):
		tokens = self.embedding(x)
		tokens = self.pos_enc(tokens)
		return self.transformer(tokens)


class PretrainHead(nn.Module):
	# 组级掩码预测头：向量化处理，接收 padding 后的索引和掩码
	def __init__(self, d_model):
		super().__init__()
		self.head = nn.Linear(d_model, 1)

	def forward(self, hidden_states, chosen_padded, chosen_valid):
		# hidden_states: (batch, num_features, d_model)
		# chosen_padded: (batch, max_group_size)  特征索引（含padding）
		# chosen_valid:  (batch, max_group_size)  有效位置掩码
		batch_size, max_group_size = chosen_padded.shape

		# gather 对应位置的 hidden：(batch, max_group_size, d_model)
		idx = chosen_padded.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
		masked_hidden = hidden_states.gather(1, idx)

		# 预测：(batch, max_group_size)
		preds = self.head(masked_hidden).squeeze(-1)
		return preds
