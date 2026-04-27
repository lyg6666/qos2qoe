# Stage2: 微调预测（支持回归/分类 + 滑动窗口）
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from config import (
	BACKBONE_CONFIG, DCN_CONFIG, FINETUNE_CONFIG, WINDOW_CONFIG,
	DEFAULT_CHECKPOINT_DIR, DEFAULT_DATASET_OUTPUT_DIR,
	TARGET_MAP, TARGET_COLS, TASK_TYPE, CLASSIFICATION_CONFIG,
)
from dataset import prepare_finetune_data
from model import FullModel
from util import read_csv, build_scheduler, get_device


def train(args):
	device = get_device()
	cfg_b = BACKBONE_CONFIG
	cfg_d = DCN_CONFIG
	cfg_t = FINETUNE_CONFIG
	target_col = TARGET_MAP[args.target]
	task_type = TASK_TYPE.get(args.target, "regression")
	seq_len = WINDOW_CONFIG["seq_len"]

	if task_type == "classification":
		clf_cfg = CLASSIFICATION_CONFIG[args.target]
		output_dim = clf_cfg["num_classes"]
		class_bins = clf_cfg["bins"]
	else:
		output_dim = 1
		class_bins = None

	# 加载预训练 checkpoint
	ckpt_dir = Path(args.ckpt_dir)
	pretrain_ckpt = torch.load(ckpt_dir / "pretrain_backbone.pt", map_location=device, weights_only=False)
	scaler_X = pretrain_ckpt["scaler_X"]

	# 加载数据
	df = read_csv(args.data_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]
	train_set, val_set, test_set, scaler_X, scaler_y = prepare_finetune_data(
		df, feature_cols, target_col, scaler_X=scaler_X,
		seq_len=seq_len, task_type=task_type, class_bins=class_bins,
	)

	train_loader = DataLoader(train_set, batch_size=cfg_t["batch_size"], shuffle=True)
	val_loader = DataLoader(val_set, batch_size=cfg_t["batch_size"])

	# 构建模型，加载预训练 backbone
	model = FullModel(
		num_features=cfg_b["num_features"], d_model=cfg_b["d_model"],
		n_heads=cfg_b["n_heads"], n_layers=cfg_b["n_layers"],
		cross_layers=cfg_d["cross_layers"], deep_dims=cfg_d["deep_dims"],
		dropout=cfg_d["dropout"],
		seq_len=seq_len, output_dim=output_dim,
	).to(device)
	model.backbone.load_state_dict(pretrain_ckpt["backbone_state_dict"])

	if cfg_t["freeze_backbone"]:
		for p in model.backbone.parameters():
			p.requires_grad = False

	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"],
	)
	scheduler = build_scheduler(optimizer, cfg_t["warmup_epochs"], cfg_t["epochs"], len(train_loader))

	if task_type == "classification":
		criterion = nn.CrossEntropyLoss()
	else:
		criterion = nn.MSELoss()

	best_val_loss = float("inf")
	patience_counter = 0
	ckpt_path = ckpt_dir / f"finetune_{args.target}.pt"

	for epoch in range(1, cfg_t["epochs"] + 1):
		model.train()
		train_loss = 0.0
		for X, y in train_loader:
			X, y = X.to(device), y.to(device)
			if task_type == "regression":
				y = y.squeeze()
			pred = model(X)
			loss = criterion(pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			train_loss += loss.item() * X.size(0)
		train_loss /= len(train_set)

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for X, y in val_loader:
				X, y = X.to(device), y.to(device)
				if task_type == "regression":
					y = y.squeeze()
				pred = model(X)
				loss = criterion(pred, y)
				val_loss += loss.item() * X.size(0)
		val_loss /= len(val_set)

		print(f"Epoch {epoch}/{cfg_t['epochs']}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			patience_counter = 0
			torch.save({
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scaler_X": scaler_X,
				"scaler_y": scaler_y,
				"epoch": epoch,
				"best_val_loss": best_val_loss,
				"config": {**cfg_b, **cfg_d},
				"target": args.target,
				"task_type": task_type,
				"seq_len": seq_len,
				"output_dim": output_dim,
				"class_bins": class_bins,
			}, ckpt_path)
			print(f"  -> saved best model (val_loss={best_val_loss:.6f})")
		else:
			patience_counter += 1
			if patience_counter >= cfg_t["patience"]:
				print(f"Early stopping at epoch {epoch}")
				break

	print(f"Finetune done. Best val_loss={best_val_loss:.6f}, saved to {ckpt_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, required=True, choices=list(TARGET_MAP.keys()))
	parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATASET_OUTPUT_DIR / "dataset.csv"))
	parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	args = parser.parse_args()
	train(args)
