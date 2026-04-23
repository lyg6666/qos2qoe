# Dataset + 数据预处理
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config import DATA_SPLIT_CONFIG, BACKBONE_CONFIG


class PretrainDataset(Dataset):
	# 预训练用：返回标准化后的68维特征，训练时在线做mask
	def __init__(self, X):
		self.X = torch.tensor(X, dtype=torch.float32)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx]


class FinetuneDataset(Dataset):
	# 微调用：返回68维特征 + 目标值
	def __init__(self, X, y):
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


def prepare_pretrain_data(
	df,
	feature_cols,
	val_ratio=DATA_SPLIT_CONFIG["val_ratio"],
	test_ratio=DATA_SPLIT_CONFIG["test_ratio"],
):
	# 预训练只需要特征，不需要目标
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
	val_ratio=DATA_SPLIT_CONFIG["val_ratio"],
	test_ratio=DATA_SPLIT_CONFIG["test_ratio"],
):
	# 微调需要特征+目标
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

	scaler_y = StandardScaler()
	y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
	y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
	y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)

	train_set = FinetuneDataset(X_train, y_train)
	val_set = FinetuneDataset(X_val, y_val)
	test_set = FinetuneDataset(X_test, y_test)
	return train_set, val_set, test_set, scaler_X, scaler_y


def mask_features(X, num_features=BACKBONE_CONFIG["num_features"]):
	# 在线mask：每个样本随机mask一个特征
	# X: (batch, num_features)
	batch_size = X.shape[0]
	mask_indices = torch.randint(0, num_features, (batch_size,))
	mask = torch.ones_like(X, dtype=torch.bool)
	mask[torch.arange(batch_size), mask_indices] = False
	targets = X[torch.arange(batch_size), mask_indices]  # 被mask的真实值
	X_masked = X.clone()
	X_masked[torch.arange(batch_size), mask_indices] = 0.0  # 用0替代（后续embedding层会用mask token）
	return X_masked, mask_indices, targets
