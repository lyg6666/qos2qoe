# ml-predict-v4

基于 Transformer + DCN 的两阶段网络 QoS→QoE 预测模型。v4 在 v3 基础上引入**组级掩码预训练**、**滑动窗口 + LSTM 时序建模**，并将 `stall_rate` 改为**分类任务**。

## 预测目标

| 目标 | 模型文件 | 任务类型 | 说明 |
|------|----------|----------|------|
| 首帧的平均 | finetune_ttfb.pt | 回归 | 首帧加载时间 |
| pwc卡顿数 | finetune_stall_count.pt | 回归 | 播放过程卡顿次数 |
| pwc卡顿率 | finetune_stall_rate.pt | 分类（4类） | 播放过程卡顿比率区间 |

`stall_rate` 分类区间：`[0,0.2)` / `[0.2,0.3)` / `[0.3,0.4)` / `[0.4,inf)`

三个目标共享同一个预训练 backbone，分别微调独立的 DCN 头。

## 项目结构

```
├── config.py            # 全部超参数配置
├── dataset.py           # 数据集（PretrainDataset 组级掩码 + FinetuneDataset / SlidingWindowFinetuneDataset）
├── model/
│   ├── backbone.py      # FeatureEmbedding + PositionalEncoding + TransformerBackbone + PretrainHead
│   ├── dcn.py           # Deep & Cross Network
│   └── __init__.py      # PretrainModel / FullModel 组装（含 LSTM 时序模块）
├── train_pretrain.py    # Stage1: 组级掩码预训练
├── train_finetune.py    # Stage2: 微调预测（支持回归/分类 + 滑动窗口）
├── eval.py              # 评估（回归: MAE/MSE/RMSE/R2 + 散点图；分类: Accuracy/F1 + 混淆矩阵）
├── predict.py           # 新数据推理
└── util.py              # 工具函数（数据合并、scheduler、可视化）
```

## 模型结构

### Stage1: Transformer Backbone（组级掩码自监督预训练）

68 维 QoS 特征 → 各自 Linear(1, 64) + 正弦位置编码生成 68 个 token → 2 层 Transformer Encoder → **随机 mask 一个语义分组**（如"握手耗时"含 10 个特征），预测该组所有特征的原始值

相比 v3 的单特征 mask，组级 mask 迫使模型学习特征间的语义关联。

### Stage2: DCN + LSTM（监督微调）

加载预训练 backbone，backbone 默认冻结。

- **单步（seq_len=1）**：68 个 token concat 为 4352 维 → 3 层 Cross Layer + Deep Network → 预测值
- **滑动窗口（seq_len=5）**：连续 5 个时间步各自过 backbone → Linear 投影到 256 维 → 单层 LSTM 提取时序特征 → DCN 预测

## 训练配置

### 预训练（Stage1）
- 优化器: Adam (lr=1e-3, weight_decay=1e-4)
- 学习率调度: Warmup 5 epochs + Cosine Annealing
- Early Stopping: patience=15
- 批大小: 512
- 最大轮数: 100

### 微调（Stage2）
- 优化器: Adam (lr=5e-5, weight_decay=1e-4)
- 学习率调度: Warmup 5 epochs + Cosine Annealing
- Early Stopping: patience=15
- 批大小: 512
- 最大轮数: 100
- backbone 默认冻结（`freeze_backbone=True`）

## 数据切分

按顺序切分（非随机），比例：train 65% / val 15% / test 20%

特征做 StandardScaler 标准化，仅在训练集上 fit；回归目标同样做标准化，分类目标直接 `np.digitize` 离散化。

## 使用方法

### 安装依赖

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### 预训练（Stage1）

```bash
python train_pretrain.py
```

自动从 `~/code/raw_dataset/` 合并数据到 `~/code/dataset/dataset.csv`，训练 backbone 并保存到 `checkpoints/pretrain_backbone.pt`。

### 微调（Stage2）

```bash
# 微调预测卡顿率（分类）
python train_finetune.py --target stall_rate

# 微调预测首帧时间（回归）
python train_finetune.py --target ttfb

# 微调预测卡顿次数（回归）
python train_finetune.py --target stall_count
```

加载预训练 backbone，微调 DCN 头，保存到 `checkpoints/finetune_<target>.pt`。

### 评估

```bash
python eval.py --target stall_rate
python eval.py --target ttfb
```

- 回归目标：输出 MAE、MSE、RMSE、R2，生成预测 vs 真实散点图
- 分类目标：输出 Accuracy、Macro-F1、混淆矩阵及分类报告，生成混淆矩阵图

结果保存到 `~/code/output/eval/<target>_<metric>_<timestamp>/`。

### 预测新数据

```bash
python predict.py --input new_server_data.csv --target stall_rate
```

## Checkpoint 内容

### pretrain_backbone.pt
- `backbone_state_dict`: backbone 权重
- `pretrain_head_state_dict`: 预训练头权重
- `scaler_X`: 特征标准化器
- `groups`: 特征分组索引
- `epoch` / `best_val_loss`: 训练信息

### finetune_\<target\>.pt
- `model_state_dict`: 完整模型权重（backbone + DCN）
- `scaler_X`: 特征标准化器
- `scaler_y`: 目标标准化器（仅回归任务）
- `task_type`: `"regression"` 或 `"classification"`
- `seq_len`: 滑动窗口长度
- `output_dim`: 输出维度（回归=1，分类=类别数）
- `class_bins`: 分类区间边界（仅分类任务）
- `epoch` / `best_val_loss`: 训练信息

## v3 → v4 主要变化

| 特性 | v3 | v4 |
|------|----|----|
| 预训练掩码策略 | 随机 mask 单个特征 | 随机 mask 语义分组（组级掩码） |
| 时序建模 | 无 | 滑动窗口（seq_len=5）+ LSTM |
| stall_rate 任务 | 回归 | 分类（4类） |
| backbone 微调 | 解冻 | 默认冻结 |
| 批大小 | 64 | 512 |
| 预训练 lr | 1e-4 | 1e-3 |
| 微调 lr | 5e-4 | 5e-5 |
