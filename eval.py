# 评估
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from config import (
	BACKBONE_CONFIG, DCN_CONFIG, EVAL_CONFIG,
	DEFAULT_CHECKPOINT_DIR, DEFAULT_DATASET_OUTPUT_DIR,
	DEFAULT_EVAL_PLOT_DIR, TARGET_MAP, TARGET_COLS,
)
from dataset import FinetuneDataset
from model import FullModel
from util import read_csv, visualize_eval_results
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_col = TARGET_MAP[args.target]

	# 加载checkpoint
	ckpt_path = Path(args.ckpt_dir) / f"finetune_{args.target}.pt"
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	cfg = ckpt["config"]
	scaler_X = ckpt["scaler_X"]
	scaler_y = ckpt["scaler_y"]

	# 加载数据，切出test集
	df = read_csv(args.data_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]

	X = df[feature_cols].values.astype(np.float32)
	y = df[target_col].values.astype(np.float32)
	log_count = df["日志数"].values.astype(np.float32)

	test_idx = int(len(X) * 0.8)
	val_idx = int(test_idx * 0.85)
	X_test = np.nan_to_num(scaler_X.transform(X[test_idx:]), nan=0.0).astype(np.float32)
	y_test = scaler_y.transform(y[test_idx:].reshape(-1, 1)).flatten().astype(np.float32)
	log_test = log_count[test_idx:]

	test_set = FinetuneDataset(X_test, y_test)
	test_loader = DataLoader(test_set, batch_size=EVAL_CONFIG["batch_size"])

	# 构建模型
	model = FullModel(
		num_features=cfg["num_features"], d_model=cfg["d_model"],
		n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
		cross_layers=cfg["cross_layers"], deep_dims=cfg.get("deep_dims", [256, 128]),
		dropout=cfg.get("dropout", 0.1),
	).to(device)
	model.load_state_dict(ckpt["model_state_dict"])
	model.eval()

	# 推理
	all_pred, all_true = [], []
	with torch.no_grad():
		for X, y in test_loader:
			X = X.to(device)
			pred = model(X)
			all_pred.append(pred.cpu().numpy())
			all_true.append(y.numpy().squeeze())

	y_pred_scaled = np.concatenate(all_pred)
	y_true_scaled = np.concatenate(all_true)

	# 反标准化
	y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
	y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

	# 指标
	mae = mean_absolute_error(y_true, y_pred)
	mse = mean_squared_error(y_true, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_true, y_pred)
	print(f"[{args.target}] MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

	# 可视化
	plot_path = visualize_eval_results(y_true, y_pred, log_test, args.target)
	print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, required=True, choices=list(TARGET_MAP.keys()))
	parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATASET_OUTPUT_DIR / "dataset.csv"))
	parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	args = parser.parse_args()
	evaluate(args)
