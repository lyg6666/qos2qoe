from model.backbone import TransformerBackbone, PretrainHead
from model.dcn import DCN
import torch.nn as nn


class PretrainModel(nn.Module):
	# Stage1: Backbone + 掩码预测头
	def __init__(self, num_features, d_model, n_heads, n_layers, dropout=0.1):
		super().__init__()
		self.backbone = TransformerBackbone(num_features, d_model, n_heads, n_layers, dropout)
		self.pretrain_head = PretrainHead(d_model)

	def forward(self, x, mask_indices):
		hidden = self.backbone(x, mask_indices)
		pred = self.pretrain_head(hidden, mask_indices)
		return pred


class FullModel(nn.Module):
	# Stage2: Backbone + DCN 预测卡顿率
	def __init__(self, num_features, d_model, n_heads, n_layers,
				 cross_layers=3, deep_dims=None, dropout=0.1):
		super().__init__()
		self.backbone = TransformerBackbone(num_features, d_model, n_heads, n_layers, dropout)
		self.dcn = DCN(num_features * d_model, cross_layers, deep_dims, dropout)
		self.num_features = num_features
		self.d_model = d_model

	def forward(self, x):
		hidden = self.backbone(x)  # (batch, num_features, d_model)
		flat = hidden.view(hidden.size(0), -1)  # (batch, num_features * d_model)
		return self.dcn(flat)  # (batch,)
