# 评估 + 预测
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from util import read_csv
from dataset import PredictDataset, prepare_data
from model import MLP

# 英文key → 中文列名映射
TARGET_MAP = {
    'ttfb': '首帧的平均',
    'stall_count': 'pwc卡顿数',
    'stall_rate': 'pwc卡顿率',
}


def load_model(model_path):
    # 从 checkpoint 加载模型 + scaler
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    input_dim = scaler_X.n_features_in_
    model = MLP(input_dim, config['hidden_dims'], config['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, scaler_X, scaler_y


def evaluate(model, test_set, scaler_y, batch_size=32):
    device = next(model.parameters()).device
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in test_dl:
            preds.append(model(X.to(device)).cpu().numpy())
            labels.append(y.numpy())

    # 反标准化
    y_pred = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(np.concatenate(labels).reshape(-1, 1)).flatten()

    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
    }

    print(f"  MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
          f"R2={metrics['R2']:.4f} | MAPE={metrics['MAPE']:.2f}%")
    return metrics, y_true, y_pred


def predict(model, df, feature_cols, scaler_X, scaler_y):
    # 新数据预测: scaler从checkpoint加载，不需要重新fit
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(scaler_X.transform(X), nan=0.0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_scaled = model(X_tensor).numpy()

    y_pred = scaler_y.inverse_transform(y_scaled).flatten()
    return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'predict'])
    args = parser.parse_args()

    model, scaler_X, scaler_y = load_model(args.model)
    df = read_csv(args.data)
    exclude = set(TARGET_MAP.values())
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]

    col = TARGET_MAP.get(args.target, args.target)

    if args.mode == 'eval':
        _, _, test_set, _, _, _ = prepare_data(df, feature_cols, col)
        print(f"Evaluate: {args.target} ({col})")
        evaluate(model, test_set, scaler_y)
    else:
        y_pred = predict(model, df, feature_cols, scaler_X, scaler_y)
        print(f"Predict: {args.target} ({col})")
        for i, v in enumerate(y_pred):
            print(f"  sample {i}: {v:.4f}")
