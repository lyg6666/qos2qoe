# 推理预测：输入新数据，输出预测结果
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
	BACKBONE_CONFIG, DCN_CONFIG, DEFAULT_CHECKPOINT_DIR,
	TARGET_MAP, TARGET_COLS, CLASSIFICATION_CONFIG,
)
from model import FullModel
from util import read_csv, get_device


def predict(args):
	device = get_device()
	target_col = TARGET_MAP[args.target]

	ckpt_path = Path(args.ckpt_dir) / f"finetune_{args.target}.pt"
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	cfg = ckpt["config"]
	scaler_X = ckpt["scaler_X"]
	scaler_y = ckpt.get("scaler_y")
	task_type = ckpt.get("task_type", "regression")
	seq_len = ckpt.get("seq_len", 1)
	output_dim = ckpt.get("output_dim", 1)

	df = read_csv(args.input_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]
	X = df[feature_cols].values.astype(np.float32)
	X = np.nan_to_num(scaler_X.transform(X), nan=0.0).astype(np.float32)

	model = FullModel(
		num_features=cfg["num_features"], d_model=cfg["d_model"],
		n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
		cross_layers=cfg["cross_layers"], deep_dims=cfg.get("deep_dims", [256, 128]),
		dropout=cfg.get("dropout", 0.1),
		seq_len=seq_len, output_dim=output_dim,
	).to(device)
	model.load_state_dict(ckpt["model_state_dict"])
	model.eval()

	if seq_len > 1:
		n_samples = len(X) - seq_len
		indices = torch.arange(n_samples).unsqueeze(1) + torch.arange(seq_len)
		X_tensor = torch.tensor(X, dtype=torch.float32)[indices].to(device)
		df_out = df.iloc[seq_len:].copy().reset_index(drop=True)
	else:
		X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
		df_out = df.copy()

	with torch.no_grad():
		pred_raw = model(X_tensor).cpu().numpy()

	if task_type == "classification":
		pred_class = pred_raw.argmax(axis=1)
		pred_prob = torch.softmax(torch.tensor(pred_raw), dim=1).numpy()

		clf_cfg = CLASSIFICATION_CONFIG[args.target]
		class_names = clf_cfg["class_names"]

		df_out[f"pred_{args.target}_class"] = pred_class
		for i, name in enumerate(class_names):
			df_out[f"prob_{name}"] = pred_prob[:, i]
	else:
		pred = scaler_y.inverse_transform(pred_raw.reshape(-1, 1)).flatten()
		df_out[f"pred_{args.target}"] = pred

	output_path = Path(args.output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
	print(f"预测完成，{len(df_out)}条结果保存到 {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, required=True, choices=list(TARGET_MAP.keys()))
	parser.add_argument("--input_path", type=str, required=True, help="输入CSV路径")
	parser.add_argument("--output_path", type=str, default="predictions.csv")
	parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	args = parser.parse_args()
	predict(args)
