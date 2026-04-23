# 推理预测：输入新数据，输出预测结果
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
	BACKBONE_CONFIG, DCN_CONFIG, DEFAULT_CHECKPOINT_DIR, TARGET_MAP, TARGET_COLS,
)
from model import FullModel
from util import read_csv


def predict(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_col = TARGET_MAP[args.target]

	# 加载checkpoint
	ckpt_path = Path(args.ckpt_dir) / f"finetune_{args.target}.pt"
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	cfg = ckpt["config"]
	scaler_X = ckpt["scaler_X"]
	scaler_y = ckpt["scaler_y"]

	# 加载数据
	df = read_csv(args.input_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]
	X = df[feature_cols].values.astype(np.float32)
	X = np.nan_to_num(scaler_X.transform(X), nan=0.0).astype(np.float32)

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
	X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
	with torch.no_grad():
		pred_scaled = model(X_tensor).cpu().numpy()
	pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

	# 输出
	df_out = df.copy()
	df_out[f"pred_{args.target}"] = pred
	output_path = Path(args.output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
	print(f"预测完成，{len(pred)}条结果保存到 {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, required=True, choices=list(TARGET_MAP.keys()))
	parser.add_argument("--input_path", type=str, required=True, help="输入CSV路径")
	parser.add_argument("--output_path", type=str, default="predictions.csv")
	parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	args = parser.parse_args()
	predict(args)
