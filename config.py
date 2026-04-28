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

# 任务类型
TASK_TYPE = {
	"ttfb": "regression",
	"stall_count": "regression",
	"stall_rate": "classification",
}

# 指标分组（用于组级 mask 预训练）
METRIC_GROUPS = {
	"日志数": ["日志数"],
	"AppLimit": ["AppLimit次数"],
	"握手耗时": [
		"握手耗时–P1", "握手耗时–P5", "握手耗时–P10", "握手耗时–P20",
		"握手耗时–P50", "握手耗时–P80", "握手耗时–P90", "握手耗时–P95",
		"握手耗时–P99", "握手耗时",
	],
	"Range": [
		"p99–Range起始值", "p5–Range起始值", "p95–Range起始值",
		"p80–Range起始值", "range大小", "range大小（小于2MB）", "Range起始值",
	],
	"用户IP": ["用户ip数目", "用户IP前缀数（省略1段）"],
	"时间戳": ["最大时间戳", "最小时间戳"],
	"慢启动": ["连接结束时未退出慢启动数", "连接结束时未退出慢启动占比"],
	"ISP": ["in_same_isp_ratio"],
	"发送内容": ["发送内容大小(包括头部)"],
	"RTT": [
		"0–RTT 数", "0RTT 占比", "early data成功次数", "early data成功占比",
		"P99Srtt", "P90SRTT", "P95SRTT", "SRTT与minRTT比例关系",
		"MinRtt", "MaxRtt", "Srtt", 
	],
	"下载速率": [
		"流速率低于BR时SRTT",
		"流速率低于BR时SRTT与minRTT比例关系",
	],
	"带宽": ["带宽max_bw", "流速率小于码率数", "流速率小于1点25倍码率数"],
	"服务器IP": ["服务器ip数目"],
	"视频观看": ["视频总观看数（视频号）", "视频平均分片数", "首分片数", "用户平均观看分片数", "用户平均观看视频数（视频号）", "非首分片数"],
	"流量": ["所有发送的字节数（含重传）", "发送字节和（流量）"],
	"重传相关": ["TLP次数", "重传字节数", "重传率", "RTO次数"],
	"下载速率区间": ["DownloadSpeed在10~500之间的数量", "DownloadSpeed在10~1000之间的数量", "DownloadSpeed在10~500之间的占比", "DownloadSpeed在10~1000之间的占比", "P1_DownloadSpeed", "P5_DownloadSpeed", "P10_DownloadSpeed", "P90_DownloadSpeed", "下载速率", "从连接建立到收到第500KB字节ACK的时间", "P50_DownloadSpeed"],
	"拥塞控制": ["拥塞控制限制时间"],
}

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

# 预训练配置（Stage1: 组级掩码预测）
PRETRAIN_CONFIG = {
	"epochs": 100,
	"warmup_epochs": 5,
	"lr": 1e-3,
	"batch_size": 256,
	"patience": 15,
	"weight_decay": 1e-4,
}

# 滑动窗口配置
WINDOW_CONFIG = {
	"seq_len": 5,
}

# 分类配置
CLASSIFICATION_CONFIG = {
	"stall_rate": {
		"bins": [0, 0.2, 0.4, float("inf")],
		"num_classes": 3,
		"class_names": ["[0,0.2)", "[0.2,0.4)", "[0.4,inf)"],
	},
}

# 微调配置（Stage2）
FINETUNE_CONFIG = {
	"epochs": 100,
	"warmup_epochs": 5,
	"lr": 1e-5,
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
