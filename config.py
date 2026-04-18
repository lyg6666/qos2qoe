from pathlib import Path


QOS2QOE_DIR = Path(__file__).resolve().parent
REPO_ROOT = QOS2QOE_DIR.parent

# 数据路径与输出路径
DEFAULT_RAW_DATA_DIR = REPO_ROOT / "raw_dataset"
DEFAULT_DATASET_OUTPUT_DIR = REPO_ROOT / "dataset"
DEFAULT_CHECKPOINT_DIR = QOS2QOE_DIR / "checkpoints"
DEFAULT_EVAL_PLOT_DIR = REPO_ROOT / "output" / "eval"

# 目标列映射
TARGET_MAP = {
	"ttfb": "首帧的平均",
	"stall_count": "pwc卡顿数",
	"stall_rate": "pwc卡顿率",
}
TARGET_COLS = list(TARGET_MAP.values())

# 训练配置
MLP_CONFIG = {
	"epochs": 100,
	"lr": 1e-3,
	"batch_size": 64,
	"patience": 15,
	"weight_decay": 1e-4,
	"hidden_dims": [128, 64],
	"dropout": 0.2,
}

# 数据切分配置
DATA_SPLIT_CONFIG = {
	"val_ratio": 0.15,
	"test_ratio": 0.2,
}

# 评估配置
EVAL_CONFIG = {
	"batch_size": 32,
	"mode": "eval",
	"plot_path": None,
}

# 数据集构建与可视化配置
MERGE_CONFIG = {
	"time_col": "时间",
	"target_cols": TARGET_COLS,
	"supported_suffixes": (".csv", ".xlsx", ".xls"),
}

VIS_CONFIG = {
	"log_col": "日志数",
	"max_points": None,
}
