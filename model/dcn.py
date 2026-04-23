# Deep & Cross Network
import torch
import torch.nn as nn


class CrossLayer(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.weight = nn.Linear(input_dim, 1, bias=False)
		self.bias = nn.Parameter(torch.zeros(input_dim))

	def forward(self, x0, x):
		# x0: 初始输入, x: 当前层输入
		xw = self.weight(x)  # (batch, 1)
		return x0 * xw + self.bias + x  # (batch, input_dim)


class DCN(nn.Module):
	def __init__(self, input_dim, cross_layers=3, deep_dims=None, dropout=0.1):
		super().__init__()
		if deep_dims is None:
			deep_dims = [256, 128]

		# Cross部分
		self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])

		# Deep部分
		deep = []
		in_dim = input_dim
		for out_dim in deep_dims:
			deep.append(nn.Linear(in_dim, out_dim))
			deep.append(nn.ReLU())
			deep.append(nn.Dropout(dropout))
			in_dim = out_dim
		self.deep = nn.Sequential(*deep)

		# 输出层: cross输出 + deep输出 -> 1
		self.output_layer = nn.Linear(input_dim + deep_dims[-1], 1)

	def forward(self, x):
		# Cross
		x0 = x
		x_cross = x
		for layer in self.cross_layers:
			x_cross = layer(x0, x_cross)

		# Deep
		x_deep = self.deep(x)

		# 拼接
		out = torch.cat([x_cross, x_deep], dim=-1)
		return self.output_layer(out).squeeze(-1)  # (batch,)
