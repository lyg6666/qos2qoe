from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

# 数据路径
DEFAULT_RAW_DATA_DIR = REPO_ROOT / "raw_dataset"
DEFAULT_DATASET_OUTPUT_DIR = REPO_ROOT / "dataset"
DEFAULT_CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
DEFAULT_EVAL_PLOT_DIR = REPO_ROOT / "output" / "eval"

# 目标列映射
TARGET_MAP = {
	"ttfb": "首帧的平均",
	"stall_count": "pwc卡顿数",
	"stall_rate": "pwc卡顿率",
}
TARGET_COLS = list(TARGET_MAP.values())

# Transformer Backbone 配置
BACKBONE_CONFIG = {
	"num_features": 68,
	"d_model": 64,
	"n_heads": 4,
	"n_layers": 2,
	"dropout": 0.1,
}

# DCN 配置
DCN_CONFIG = {
	"cross_layers": 3,
	"deep_dims": [256, 128],
	"dropout": 0.1,
}

# 预训练配置（Stage1: 掩码预测）
PRETRAIN_CONFIG = {
	"epochs": 100,
	"warmup_epochs": 5,
	"lr": 1e-3,
	"batch_size": 256,
	"patience": 15,
	"weight_decay": 1e-4,
}

# 微调配置（Stage2: 预测卡顿率）
FINETUNE_CONFIG = {
	"epochs": 100,
	"warmup_epochs": 5,
	"lr": 1e-4,
	"batch_size": 256,
	"patience": 15,
	"weight_decay": 1e-4,
	"freeze_backbone": False,
}

# 数据切分配置
DATA_SPLIT_CONFIG = {
	"val_ratio": 0.15,
	"test_ratio": 0.2,
}

# 评估配置
EVAL_CONFIG = {
	"batch_size": 32,
}

# 数据集构建配置
MERGE_CONFIG = {
	"time_col": "时间",
	"target_cols": TARGET_COLS,
	"supported_suffixes": (".csv", ".xlsx", ".xls"),
}

VIS_CONFIG = {
	"log_col": "日志数",
	"max_points": None,
}
