# Stage1: 掩码预训练 Backbone
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from config import (
	BACKBONE_CONFIG, PRETRAIN_CONFIG, DEFAULT_CHECKPOINT_DIR,
	DEFAULT_DATASET_OUTPUT_DIR, TARGET_MAP, TARGET_COLS,
)
from dataset import prepare_pretrain_data, mask_features
from model import PretrainModel
from util import read_csv, build_scheduler


def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cfg_b = BACKBONE_CONFIG
	cfg_t = PRETRAIN_CONFIG

	# 加载数据
	df = read_csv(args.data_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]
	train_set, val_set, scaler_X = prepare_pretrain_data(df, feature_cols)

	train_loader = DataLoader(train_set, batch_size=cfg_t["batch_size"], shuffle=True)
	val_loader = DataLoader(val_set, batch_size=cfg_t["batch_size"])

	# 模型
	model = PretrainModel(
		num_features=cfg_b["num_features"], d_model=cfg_b["d_model"],
		n_heads=cfg_b["n_heads"], n_layers=cfg_b["n_layers"],
		dropout=cfg_b["dropout"],
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])
	scheduler = build_scheduler(optimizer, cfg_t["warmup_epochs"], cfg_t["epochs"], len(train_loader))
	criterion = nn.MSELoss()

	best_val_loss = float("inf")
	patience_counter = 0
	ckpt_dir = Path(args.ckpt_dir)
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = ckpt_dir / "pretrain_backbone.pt"

	for epoch in range(1, cfg_t["epochs"] + 1):
		# Train
		model.train()
		train_loss = 0.0
		for X in train_loader:
			X = X.to(device)
			X_masked, mask_indices, targets = mask_features(X)
			pred = model(X_masked, mask_indices.to(device))
			loss = criterion(pred, targets.to(device))
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
			for X in val_loader:
				X = X.to(device)
				X_masked, mask_indices, targets = mask_features(X)
				pred = model(X_masked, mask_indices.to(device))
				loss = criterion(pred, targets.to(device))
				val_loss += loss.item() * X.size(0)
		val_loss /= len(val_set)

		print(f"Epoch {epoch}/{cfg_t['epochs']}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			patience_counter = 0
			torch.save({
				"backbone_state_dict": model.backbone.state_dict(),
				"pretrain_head_state_dict": model.pretrain_head.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scaler_X": scaler_X,
				"epoch": epoch,
				"best_val_loss": best_val_loss,
				"config": cfg_b,
			}, ckpt_path)
			print(f"  -> saved best model (val_loss={best_val_loss:.6f})")
		else:
			patience_counter += 1
			if patience_counter >= cfg_t["patience"]:
				print(f"Early stopping at epoch {epoch}")
				break

	print(f"Pretrain done. Best val_loss={best_val_loss:.6f}, saved to {ckpt_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATASET_OUTPUT_DIR / "dataset.csv"))
	parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	args = parser.parse_args()
	train(args)
