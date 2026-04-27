# Deep & Cross Network
import torch
import torch.nn as nn


class CrossLayer(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.weight = nn.Linear(input_dim, 1, bias=False)
		self.bias = nn.Parameter(torch.zeros(input_dim))

	def forward(self, x0, x):
		xw = self.weight(x)
		return x0 * xw + self.bias + x


class DCN(nn.Module):
	def __init__(self, input_dim, cross_layers=3, deep_dims=None, dropout=0.1, output_dim=1):
		super().__init__()
		if deep_dims is None:
			deep_dims = [256, 128]

		self.output_dim = output_dim
		self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])

		deep = []
		in_dim = input_dim
		for out_dim in deep_dims:
			deep.append(nn.Linear(in_dim, out_dim))
			deep.append(nn.ReLU())
			deep.append(nn.Dropout(dropout))
			in_dim = out_dim
		self.deep = nn.Sequential(*deep)

		self.output_layer = nn.Linear(input_dim + deep_dims[-1], output_dim)

	def forward(self, x):
		x0 = x
		x_cross = x
		for layer in self.cross_layers:
			x_cross = layer(x0, x_cross)

		x_deep = self.deep(x)
		out = torch.cat([x_cross, x_deep], dim=-1)
		out = self.output_layer(out)
		if self.output_dim == 1:
			return out.squeeze(-1)
		return out
