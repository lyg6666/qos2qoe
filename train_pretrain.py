# Stage1: 组级掩码预训练 Backbone
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from config import (
	BACKBONE_CONFIG, PRETRAIN_CONFIG, DEFAULT_CHECKPOINT_DIR,
	DEFAULT_DATASET_OUTPUT_DIR, TARGET_COLS,
)
from dataset import prepare_pretrain_data, build_group_indices, GroupMasker
from model import PretrainModel
from util import read_csv, build_scheduler, get_device


def group_mse_loss(preds, targets_pad, chosen_valid):
	diff = (preds - targets_pad) ** 2
	loss = (diff * chosen_valid.float()).sum() / chosen_valid.float().sum()
	return loss


def train(args):
	device = get_device()
	print(f"Using device: {device}")
	cfg_b = BACKBONE_CONFIG
	cfg_t = PRETRAIN_CONFIG

	df = read_csv(args.data_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]
	train_set, val_set, scaler_X = prepare_pretrain_data(df, feature_cols)
	groups = build_group_indices(feature_cols)
	masker = GroupMasker(groups, device=device)

	train_loader = DataLoader(train_set, batch_size=cfg_t["batch_size"], shuffle=True)
	val_loader = DataLoader(val_set, batch_size=cfg_t["batch_size"])

	model = PretrainModel(
		groups=groups,
		d_model=cfg_b["d_model"],
		n_heads=cfg_b["n_heads"],
		n_layers=cfg_b["n_layers"],
		dropout=cfg_b["dropout"],
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])
	scheduler = build_scheduler(optimizer, cfg_t["warmup_epochs"], cfg_t["epochs"], len(train_loader))

	best_val_loss = float("inf")
	patience_counter = 0
	ckpt_dir = Path(args.ckpt_dir)
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = ckpt_dir / "pretrain_backbone.pt"

	for epoch in range(1, cfg_t["epochs"] + 1):
		model.train()
		train_loss = 0.0
		train_batches = 0
		for X in train_loader:
			X = X.to(device)
			chosen_group_idx, targets_pad, chosen_valid = masker.mask(X)
			preds, valid = model(X, chosen_group_idx)
			loss = group_mse_loss(preds, targets_pad, chosen_valid)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			train_loss += loss.item()
			train_batches += 1
		train_loss /= train_batches

		model.eval()
		val_loss = 0.0
		val_batches = 0
		with torch.no_grad():
			for X in val_loader:
				X = X.to(device)
				chosen_group_idx, targets_pad, chosen_valid = masker.mask(X)
				preds, valid = model(X, chosen_group_idx)
				loss = group_mse_loss(preds, targets_pad, chosen_valid)
				val_loss += loss.item()
				val_batches += 1
		val_loss /= val_batches

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
				"groups": groups,
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
