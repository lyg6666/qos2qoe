# ml-predict-v3

基于 Transformer + DCN 的两阶段网络 QoS→QoE 预测模型。Stage1 自监督掩码预训练学习特征表示，Stage2 微调预测视频流媒体质量指标。

## 预测目标

| 目标 | 模型文件 | 说明 |
|------|----------|------|
| 首帧的平均 | finetune_ttfb.pt | 首帧加载时间 |
| pwc卡顿数 | finetune_stall_count.pt | 播放过程卡顿次数 |
| pwc卡顿率 | finetune_stall_rate.pt | 播放过程卡顿比率 |

三个目标共享同一个预训练 backbone，分别微调独立的 DCN 头。

## 项目结构

```
├── config.py            # 全部超参数配置
├── dataset.py           # 数据集（PretrainDataset 掩码 + FinetuneDataset 监督）
├── model/
│   ├── backbone.py      # FeatureEmbedding + TransformerBackbone + PretrainHead
│   ├── dcn.py           # Deep & Cross Network
│   └── __init__.py      # PretrainModel / FullModel 组装
├── train_pretrain.py    # Stage1: 掩码预训练
├── train_finetune.py    # Stage2: 微调预测
├── eval.py              # 评估（MAE/MSE/RMSE/R2 + 散点图）
├── predict.py           # 新数据推理
├── util.py              # 工具函数（数据合并、scheduler、可视化）
└── requirements.txt
```

## 模型结构

### Stage1: Transformer Backbone（自监督预训练）

68 维 QoS 特征 → 各自 Linear(1, 64) 生成 68 个 token → 2 层 Transformer Encoder → 随机 mask 一个特征，预测其原始值

### Stage2: DCN（监督微调）

加载预训练 backbone → 68 个 token concat 为 4352 维 → 3 层 Cross Layer + Deep Network → 预测 QoE 指标

## 训练配置

### 预训练（Stage1）
- 优化器: Adam (lr=1e-4, weight_decay=1e-4)
- 学习率调度: Warmup 5 epochs + Cosine Annealing
- Early Stopping: patience=10
- 批大小: 64
- 最大轮数: 200

### 微调（Stage2）
- 优化器: Adam (lr=5e-4, weight_decay=1e-4)
- 学习率调度: Warmup 5 epochs + Cosine Annealing
- Early Stopping: patience=10
- 批大小: 64
- 最大轮数: 100

## 数据切分

按顺序切分（非随机），比例：train 65% / val 15% / test 20%

特征做 StandardScaler 标准化，仅在训练集上 fit。

## 使用方法

### 预训练（Stage1）

```bash
python train_pretrain.py
```

自动从 `~/code/raw_dataset/` 合并数据到 `~/code/dataset/dataset.csv`，训练 backbone 并保存到 `checkpoints/pretrain_backbone.pt`。

### 微调（Stage2）

```bash
# 微调预测卡顿率
python train_finetune.py --target stall_rate

# 微调预测首帧时间
python train_finetune.py --target ttfb
```

加载预训练 backbone，微调 DCN 头，保存到 `checkpoints/finetune_<target>.pt`。

### 评估

```bash
python eval.py --target stall_rate
```

输出指标：MAE、MSE、RMSE、R2，并生成散点图到 `~/code/output/eval/`。

### 预测新数据

```bash
python predict.py --input new_server_data.csv --target stall_rate
```

## Checkpoint 内容

### pretrain_backbone.pt
- model_state_dict: backbone 权重
- epoch / best_val_loss: 训练信息

### finetune_<target>.pt
- model_state_dict: 完整模型权重（backbone + DCN）
- scaler_X: 特征标准化器
- epoch / best_val_loss: 训练信息

## 依赖

```bash
pip install -r requirements.txt
```
