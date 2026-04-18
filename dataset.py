# Dataset + 数据预处理
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config import DATA_SPLIT_CONFIG


class PredictDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(
    df,
    feature_cols,
    target_col,
    val_ratio=DATA_SPLIT_CONFIG["val_ratio"],
    test_ratio=DATA_SPLIT_CONFIG["test_ratio"],
):
    # 提取特征和目标
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # 切分: train / val / test
    test_idx = int(len(X) * (1 - test_ratio))
    val_idx = int(test_idx * (1 - val_ratio))
    X_train, X_val, X_test = X[:val_idx], X[val_idx:test_idx], X[test_idx:]
    y_train, y_val, y_test = y[:val_idx], y[val_idx:test_idx], y[test_idx:]

    # 特征标准化 (fit on train only)
    scaler_X = StandardScaler()
    X_train = np.nan_to_num(scaler_X.fit_transform(X_train), nan=0.0).astype(np.float32)
    X_val = np.nan_to_num(scaler_X.transform(X_val), nan=0.0).astype(np.float32)
    X_test = np.nan_to_num(scaler_X.transform(X_test), nan=0.0).astype(np.float32)

    # 目标标准化
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)

    train_set = PredictDataset(X_train, y_train)
    val_set = PredictDataset(X_val, y_val)
    test_set = PredictDataset(X_test, y_test)
    return train_set, val_set, test_set, X_train.shape[1], scaler_X, scaler_y
