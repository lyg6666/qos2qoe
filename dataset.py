# Dataset + 数据预处理
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config import DATA_SPLIT_CONFIG, BACKBONE_CONFIG, METRIC_GROUPS


class PretrainDataset(Dataset):
	def __init__(self, X):
		self.X = torch.tensor(X, dtype=torch.float32)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx]


class FinetuneDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


class SlidingWindowFinetuneDataset(Dataset):
	def __init__(self, X, y, seq_len, is_classification=False):
		self.seq_len = seq_len
		n_samples = len(X) - seq_len
		self.X = torch.zeros(n_samples, seq_len, X.shape[1], dtype=torch.float32)
		for i in range(n_samples):
			self.X[i] = torch.tensor(X[i:i + seq_len], dtype=torch.float32)

		y_target = y[seq_len:]
		if is_classification:
			self.y = torch.tensor(y_target, dtype=torch.long)
		else:
			self.y = torch.tensor(y_target, dtype=torch.float32).unsqueeze(1)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


def build_group_indices(feature_cols, metric_groups=METRIC_GROUPS):
	col_to_idx = {col: i for i, col in enumerate(feature_cols)}
	groups = []
	assigned = set()

	for group_name, col_names in metric_groups.items():
		indices = []
		for col in col_names:
			if col in col_to_idx:
				indices.append(col_to_idx[col])
				assigned.add(col)
		if indices:
			groups.append(indices)

	# 未归类特征各自成为单元素组
	for col in feature_cols:
		if col not in assigned:
			groups.append([col_to_idx[col]])

	return groups


def build_group_mask_matrix(groups, num_features):
	"""构建 (num_groups, num_features) 布尔矩阵，每行标记该组覆盖的特征位置。"""
	num_groups = len(groups)
	matrix = torch.zeros(num_groups, num_features, dtype=torch.bool)
	for i, positions in enumerate(groups):
		matrix[i, positions] = True
	return matrix


def build_group_pad_tensors(groups):
	"""
	将各组 padding 到相同长度，返回：
	  padded_indices: (num_groups, max_group_size)  填充值为 0
	  valid_mask:     (num_groups, max_group_size)  True=有效位置
	"""
	max_len = max(len(g) for g in groups)
	num_groups = len(groups)
	padded = torch.zeros(num_groups, max_len, dtype=torch.long)
	valid = torch.zeros(num_groups, max_len, dtype=torch.bool)
	for i, g in enumerate(groups):
		padded[i, :len(g)] = torch.tensor(g, dtype=torch.long)
		valid[i, :len(g)] = True
	return padded, valid


def mask_group_features(X, groups, group_mask_matrix, padded_indices, valid_mask):
	"""
	向量化组级掩码，无 Python batch 循环。
	返回:
	  X_masked:      (batch, num_features)
	  chosen_padded: (batch, max_group_size)  选中组的特征索引（含padding）
	  targets_pad:   (batch, max_group_size)  对应原始值（padding位置为0）
	  chosen_valid:  (batch, max_group_size)  有效位置掩码
	"""
	batch_size, num_features = X.shape
	num_groups = len(groups)
	chosen = torch.randint(0, num_groups, (batch_size,), device=X.device)

	# 掩码置零：(batch, num_features)
	batch_mask = group_mask_matrix[chosen]          # (batch, num_features)
	X_masked = X.masked_fill(batch_mask, 0.0)

	# 提取 targets：用 padded_indices 批量 gather
	chosen_padded = padded_indices[chosen]           # (batch, max_group_size)
	chosen_valid = valid_mask[chosen]                # (batch, max_group_size)
	# gather 原始值
	targets_pad = X.gather(1, chosen_padded)         # (batch, max_group_size)

	return X_masked, chosen_padded, targets_pad, chosen_valid


def prepare_pretrain_data(
	df,
	feature_cols,
	val_ratio=DATA_SPLIT_CONFIG["val_ratio"],
	test_ratio=DATA_SPLIT_CONFIG["test_ratio"],
):
	X = df[feature_cols].values.astype(np.float32)

	test_idx = int(len(X) * (1 - test_ratio))
	val_idx = int(test_idx * (1 - val_ratio))
	X_train, X_val = X[:val_idx], X[val_idx:test_idx]

	scaler_X = StandardScaler()
	X_train = np.nan_to_num(scaler_X.fit_transform(X_train), nan=0.0).astype(np.float32)
	X_val = np.nan_to_num(scaler_X.transform(X_val), nan=0.0).astype(np.float32)

	return PretrainDataset(X_train), PretrainDataset(X_val), scaler_X


def prepare_finetune_data(
	df,
	feature_cols,
	target_col,
	scaler_X=None,
	seq_len=1,
	task_type="regression",
	class_bins=None,
	val_ratio=DATA_SPLIT_CONFIG["val_ratio"],
	test_ratio=DATA_SPLIT_CONFIG["test_ratio"],
):
	X = df[feature_cols].values.astype(np.float32)
	y = df[target_col].values.astype(np.float32)

	test_idx = int(len(X) * (1 - test_ratio))
	val_idx = int(test_idx * (1 - val_ratio))
	X_train, X_val, X_test = X[:val_idx], X[val_idx:test_idx], X[test_idx:]
	y_train, y_val, y_test = y[:val_idx], y[val_idx:test_idx], y[test_idx:]

	if scaler_X is None:
		scaler_X = StandardScaler()
		X_train = np.nan_to_num(scaler_X.fit_transform(X_train), nan=0.0).astype(np.float32)
	else:
		X_train = np.nan_to_num(scaler_X.transform(X_train), nan=0.0).astype(np.float32)
	X_val = np.nan_to_num(scaler_X.transform(X_val), nan=0.0).astype(np.float32)
	X_test = np.nan_to_num(scaler_X.transform(X_test), nan=0.0).astype(np.float32)

	scaler_y = None
	if task_type == "classification":
		# bins 的内部边界做 digitize，例如 [0, 0.2, 0.3, 0.4, inf] -> 内部边界 [0.2, 0.3, 0.4]
		inner_bins = class_bins[1:-1]
		y_train = np.digitize(y_train, inner_bins).astype(np.int64)
		y_val = np.digitize(y_val, inner_bins).astype(np.int64)
		y_test = np.digitize(y_test, inner_bins).astype(np.int64)
	else:
		scaler_y = StandardScaler()
		y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
		y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
		y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)

	is_clf = (task_type == "classification")
	if seq_len > 1:
		train_set = SlidingWindowFinetuneDataset(X_train, y_train, seq_len, is_clf)
		val_set = SlidingWindowFinetuneDataset(X_val, y_val, seq_len, is_clf)
		test_set = SlidingWindowFinetuneDataset(X_test, y_test, seq_len, is_clf)
	else:
		train_set = FinetuneDataset(X_train, y_train)
		val_set = FinetuneDataset(X_val, y_val)
		test_set = FinetuneDataset(X_test, y_test)

	return train_set, val_set, test_set, scaler_X, scaler_y
