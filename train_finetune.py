# Stage2: 微调 DCN 预测卡顿率
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from config import (
	BACKBONE_CONFIG, DCN_CONFIG, FINETUNE_CONFIG,
	DEFAULT_CHECKPOINT_DIR, DEFAULT_DATASET_OUTPUT_DIR,
	TARGET_MAP, TARGET_COLS,
)
from dataset import prepare_finetune_data
from model import FullModel
from util import read_csv, build_scheduler


def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cfg_b = BACKBONE_CONFIG
	cfg_d = DCN_CONFIG
	cfg_t = FINETUNE_CONFIG
	target_col = TARGET_MAP[args.target]

	# 加载预训练checkpoint
	ckpt_dir = Path(args.ckpt_dir)
	pretrain_ckpt = torch.load(ckpt_dir / "pretrain_backbone.pt", map_location=device, weights_only=False)
	scaler_X = pretrain_ckpt["scaler_X"]

	# 加载数据，使用预训练的scaler_X
	df = read_csv(args.data_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]
	train_set, val_set, test_set, scaler_X, scaler_y = prepare_finetune_data(
		df, feature_cols, target_col, scaler_X=scaler_X,
	)

	train_loader = DataLoader(train_set, batch_size=cfg_t["batch_size"], shuffle=True)
	val_loader = DataLoader(val_set, batch_size=cfg_t["batch_size"])

	# 构建完整模型，加载预训练backbone权重
	model = FullModel(
		num_features=cfg_b["num_features"], d_model=cfg_b["d_model"],
		n_heads=cfg_b["n_heads"], n_layers=cfg_b["n_layers"],
		cross_layers=cfg_d["cross_layers"], deep_dims=cfg_d["deep_dims"],
		dropout=cfg_d["dropout"],
	).to(device)
	model.backbone.load_state_dict(pretrain_ckpt["backbone_state_dict"])

	# 是否冻结backbone
	if cfg_t["freeze_backbone"]:
		for p in model.backbone.parameters():
			p.requires_grad = False

	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"],
	)
	scheduler = build_scheduler(optimizer, cfg_t["warmup_epochs"], cfg_t["epochs"], len(train_loader))
	criterion = nn.MSELoss()

	best_val_loss = float("inf")
	patience_counter = 0
	ckpt_path = ckpt_dir / f"finetune_{args.target}.pt"

	for epoch in range(1, cfg_t["epochs"] + 1):
		# Train
		model.train()
		train_loss = 0.0
		for X, y in train_loader:
			X, y = X.to(device), y.to(device).squeeze()
			pred = model(X)
			loss = criterion(pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			train_loss += loss.item() * X.size(0)
		train_loss /= len(train_set)

		# Val
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for X, y in val_loader:
				X, y = X.to(device), y.to(device).squeeze()
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
