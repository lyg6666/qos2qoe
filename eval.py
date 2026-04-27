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
	CLASSIFICATION_CONFIG,
)
from dataset import FinetuneDataset, SlidingWindowFinetuneDataset
from model import FullModel
from util import read_csv, visualize_eval_results, get_device
from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, r2_score,
	accuracy_score, f1_score, confusion_matrix, classification_report,
)


def evaluate(args):
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
	class_bins = ckpt.get("class_bins")

	df = read_csv(args.data_path)
	feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "时间"]

	X = df[feature_cols].values.astype(np.float32)
	y = df[target_col].values.astype(np.float32)
	log_count = df["日志数"].values.astype(np.float32)

	test_idx = int(len(X) * 0.8)
	X_test = np.nan_to_num(scaler_X.transform(X[test_idx:]), nan=0.0).astype(np.float32)
	y_test_raw = y[test_idx:]
	log_test = log_count[test_idx:]

	if task_type == "classification":
		inner_bins = class_bins[1:-1]
		y_test = np.digitize(y_test_raw, inner_bins).astype(np.int64)
	else:
		y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

	if seq_len > 1:
		is_clf = (task_type == "classification")
		test_set = SlidingWindowFinetuneDataset(X_test, y_test, seq_len, is_clf)
		log_test = log_test[seq_len:]
		y_test_raw = y_test_raw[seq_len:]
	else:
		test_set = FinetuneDataset(X_test, y_test)

	test_loader = DataLoader(test_set, batch_size=EVAL_CONFIG["batch_size"])

	model = FullModel(
		num_features=cfg["num_features"], d_model=cfg["d_model"],
		n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
		cross_layers=cfg["cross_layers"], deep_dims=cfg.get("deep_dims", [256, 128]),
		dropout=cfg.get("dropout", 0.1),
		seq_len=seq_len, output_dim=output_dim,
	).to(device)
	model.load_state_dict(ckpt["model_state_dict"])
	model.eval()

	all_pred, all_true = [], []
	with torch.no_grad():
		for X_batch, y_batch in test_loader:
			X_batch = X_batch.to(device)
			pred = model(X_batch)
			all_pred.append(pred.cpu().numpy())
			all_true.append(y_batch.numpy().squeeze())

	y_pred_raw = np.concatenate(all_pred)
	y_true_raw = np.concatenate(all_true)

	from datetime import datetime
	timestamp = datetime.now().strftime("%m%d_%H%M%S")

	if task_type == "classification":
		y_pred_class = y_pred_raw.argmax(axis=1)
		y_true_class = y_true_raw

		acc = accuracy_score(y_true_class, y_pred_class)
		f1_macro = f1_score(y_true_class, y_pred_class, average="macro", zero_division=0)
		cm = confusion_matrix(y_true_class, y_pred_class)

		clf_cfg = CLASSIFICATION_CONFIG[args.target]
		class_names = clf_cfg["class_names"]

		print(f"[{args.target}] Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}")
		print(f"Confusion Matrix:\n{cm}")
		print(classification_report(y_true_class, y_pred_class, target_names=class_names, zero_division=0))

		run_dir = DEFAULT_EVAL_PLOT_DIR / f"{args.target}_Acc={acc:.4f}_{timestamp}"
		run_dir.mkdir(parents=True, exist_ok=True)

		# confusion matrix 可视化
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots(figsize=(6, 5))
		im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
		ax.figure.colorbar(im, ax=ax)
		ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
			   xticklabels=class_names, yticklabels=class_names,
			   xlabel="Predicted", ylabel="True",
			   title=f"{args.target} Confusion Matrix (Acc={acc:.4f})")
		for i in range(len(class_names)):
			for j in range(len(class_names)):
				ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
						color="white" if cm[i, j] > cm.max() / 2 else "black")
		fig.tight_layout()
		plot_path = run_dir / "confusion_matrix.png"
		fig.savefig(plot_path, dpi=150)
		plt.close(fig)
		print(f"Plot saved to {plot_path}")

		info_path = run_dir / "config.txt"
		with open(info_path, "w") as f:
			f.write(f"target: {args.target}\n")
			f.write(f"task_type: {task_type}\n")
			f.write(f"Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}\n\n")
			f.write(f"Confusion Matrix:\n{cm}\n\n")
			f.write("--- model config ---\n")
			for k, v in cfg.items():
				f.write(f"  {k}: {v}\n")
	else:
		y_pred = scaler_y.inverse_transform(y_pred_raw.reshape(-1, 1)).flatten()
		y_true = scaler_y.inverse_transform(y_true_raw.reshape(-1, 1)).flatten()

		mae = mean_absolute_error(y_true, y_pred)
		mse = mean_squared_error(y_true, y_pred)
		rmse = np.sqrt(mse)
		r2 = r2_score(y_true, y_pred)
		print(f"[{args.target}] MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

		run_dir = DEFAULT_EVAL_PLOT_DIR / f"{args.target}_R2={r2:.4f}_{timestamp}"
		run_dir.mkdir(parents=True, exist_ok=True)

		plot_path = run_dir / "pred_vs_true.png"
		visualize_eval_results(y_true, y_pred, log_test, args.target, save_path=plot_path)
		print(f"Plot saved to {plot_path}")

		info_path = run_dir / "config.txt"
		with open(info_path, "w") as f:
			f.write(f"target: {args.target}\n")
			f.write(f"MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}\n\n")
			f.write("--- model config ---\n")
			for k, v in cfg.items():
				f.write(f"  {k}: {v}\n")
			f.write("\n--- finetune config ---\n")
			from config import FINETUNE_CONFIG
			for k, v in FINETUNE_CONFIG.items():
				f.write(f"  {k}: {v}\n")

	print(f"Config saved to {info_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, required=True, choices=list(TARGET_MAP.keys()))
	parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATASET_OUTPUT_DIR / "dataset.csv"))
	parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	args = parser.parse_args()
	evaluate(args)
