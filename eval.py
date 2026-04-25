# 评估 + 预测
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from util import (
    datasets_construction,
    visualize_eval_results,
    build_test_set_with_checkpoint_scalers,
    explain_qoe_change_with_shap,
)
from MLP_model import MLP
from config import TARGET_MAP, EVAL_CONFIG, DEFAULT_RAW_DATA_DIR, SHAP_CONFIG


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
    parser.add_argument('--raw_data_folder', type=str, default=str(DEFAULT_RAW_DATA_DIR))
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--mode', type=str, default=EVAL_CONFIG['mode'], choices=['eval', 'predict', 'explain'])
    parser.add_argument('--plot_path', type=str, default=EVAL_CONFIG['plot_path'])
    parser.add_argument('--sample_index', type=int, default=SHAP_CONFIG['sample_index'])
    parser.add_argument('--window_minutes', type=int, default=SHAP_CONFIG['window_minutes'])
    parser.add_argument('--top_k', type=int, default=SHAP_CONFIG['top_k'])
    parser.add_argument('--background_size', type=int, default=SHAP_CONFIG['background_size'])
    args = parser.parse_args()

    model, scaler_X, scaler_y = load_model(args.model)
    _, _, dataset = datasets_construction(rawdataset_dir=args.raw_data_folder)
    exclude = set(TARGET_MAP.values())
    feature_cols = [c for c in dataset.select_dtypes(include='number').columns if c not in exclude]
    col = TARGET_MAP.get(args.target, args.target)

    if args.mode == 'eval':
        #拆出测试机，还原test_set变换，并获取日志数
        test_set, log_test = build_test_set_with_checkpoint_scalers(dataset, feature_cols, col, scaler_X, scaler_y)
        print(f"Evaluate: {args.target} ({col})")
        #评估
        _, y_true, y_pred = evaluate(model, test_set, scaler_y, batch_size=EVAL_CONFIG['batch_size'])
        #画图
        fig_path = visualize_eval_results(y_true, y_pred, log_test, args.target, save_path=args.plot_path)
    elif args.mode == 'predict':
        y_pred = predict(model, dataset, feature_cols, scaler_X, scaler_y)
        print(f"Predict: {args.target} ({col})")
        for i, v in enumerate(y_pred):
            print(f"  sample {i}: {v:.4f}")
    elif args.mode == 'explain':
        print(f"Explain: {args.target} ({col}) | sample_index={args.sample_index}")
        result = explain_qoe_change_with_shap(
            model=model,
            df=dataset,
            feature_cols=feature_cols,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            target_name=col,
            sample_index=args.sample_index,
            window_minutes=args.window_minutes,
            top_k=args.top_k,
            background_size=args.background_size,
        )
        print(result["message"])
