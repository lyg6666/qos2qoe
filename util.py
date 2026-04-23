# 工具函数
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from config import (
	DEFAULT_RAW_DATA_DIR, DEFAULT_DATASET_OUTPUT_DIR,
	DEFAULT_EVAL_PLOT_DIR, MERGE_CONFIG, VIS_CONFIG,
)


def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
	from torch.optim.lr_scheduler import LambdaLR
	warmup_steps = warmup_epochs * steps_per_epoch
	total_steps = total_epochs * steps_per_epoch
	import math
	def lr_lambda(step):
		if step < warmup_steps:
			return step / max(warmup_steps, 1)
		progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
		return 0.5 * (1 + math.cos(math.pi * progress))
	return LambdaLR(optimizer, lr_lambda)


def read_csv(path):
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"数据文件不存在: {p}")
	return pd.read_csv(p)


def datasets_construction(
	rawdataset_dir=DEFAULT_RAW_DATA_DIR,
	output_dir=DEFAULT_DATASET_OUTPUT_DIR,
	time_col=MERGE_CONFIG["time_col"],
):
	raw_dir = Path(rawdataset_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	supported_suffixes = set(MERGE_CONFIG["supported_suffixes"])
	files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_suffixes]
	client_files = sorted([p for p in files if "client" in p.stem.lower()])
	server_files = sorted([p for p in files if "server" in p.stem.lower()])

	def _read_table(path):
		if path.suffix.lower() == ".csv":
			return pd.read_csv(path)
		return pd.read_excel(path)

	def _dedupe_by_time(df):
		value_cols = [c for c in df.columns if c != time_col]
		temp = df.copy()
		temp["__filled"] = temp[value_cols].notna().sum(axis=1)
		keep = temp.groupby(time_col, sort=False)["__filled"].idxmax()
		return temp.loc[keep].drop(columns=["__filled"]).sort_values(time_col).reset_index(drop=True)

	def _concat(file_list):
		dfs = []
		for fp in file_list:
			d = _read_table(fp)
			d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
			d = d.dropna(subset=[time_col])
			dfs.append(d)
		return _dedupe_by_time(pd.concat(dfs, ignore_index=True))

	client_all = _concat(client_files)
	server_all = _concat(server_files)
	common_times = pd.Index(client_all[time_col]).intersection(pd.Index(server_all[time_col]))

	client_f = client_all[client_all[time_col].isin(common_times)].pipe(_dedupe_by_time)
	server_f = server_all[server_all[time_col].isin(common_times)].pipe(_dedupe_by_time)

	target_cols = MERGE_CONFIG["target_cols"]
	merged = pd.concat([
		client_f[[time_col] + target_cols].reset_index(drop=True),
		server_f.drop(columns=[time_col], errors="ignore").reset_index(drop=True),
	], axis=1)

	output_paths = {"client": output_dir / "client.csv", "server": output_dir / "server.csv", "merged": output_dir / "dataset.csv"}
	client_f.to_csv(output_paths["client"], index=False, encoding="utf-8-sig")
	server_f.to_csv(output_paths["server"], index=False, encoding="utf-8-sig")
	merged.to_csv(output_paths["merged"], index=False, encoding="utf-8-sig")
	return client_f, server_f, merged


def visualize_eval_results(
	y_true, y_pred, log_count, target_name,
	save_path=None, max_points=VIS_CONFIG["max_points"],
):
	y_true = np.asarray(y_true).reshape(-1)
	y_pred = np.asarray(y_pred).reshape(-1)
	log_count = np.asarray(log_count).reshape(-1)

	if save_path is None:
		save_path = DEFAULT_EVAL_PLOT_DIR / f"{target_name}_pred_vs_true.png"
	else:
		save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	show_n = len(y_true) if max_points is None else min(len(y_true), max_points)
	idx = np.arange(show_n)

	fig, ax_left = plt.subplots(1, 1, figsize=(14, 5))
	ax_right = ax_left.twinx()

	line_true = ax_left.plot(idx, y_true[:show_n], label="True", linewidth=1.6, color="#1f77b4")
	line_pred = ax_left.plot(idx, y_pred[:show_n], label="Pred", linewidth=1.2, color="#ff7f0e")
	ax_left.set_xlabel("Sample Index")
	ax_left.set_ylabel(f"{target_name} Value")
	ax_left.grid(alpha=0.25)

	line_log = ax_right.plot(idx, log_count[:show_n], label="Log Count", linewidth=1.0, color="#2ca02c", alpha=0.75)
	ax_right.set_ylabel("log count")

	all_lines = line_true + line_pred + line_log
	ax_left.legend(all_lines, [l.get_label() for l in all_lines], loc="upper right")
	ax_left.set_title(f"{target_name}: True/Pred + Log Count (First {show_n})")

	fig.tight_layout()
	fig.savefig(save_path, dpi=150)
	plt.close(fig)
	return save_path
