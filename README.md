# ml-predict-v2

基于 MLP 的网络 QoS 指标预测模型，通过网络参数预测视频流媒体的质量指标。

## 预测目标

| 目标 | 模型文件 | 说明 |
|------|----------|------|
| 首帧的平均 | mlp_ttfb.pt | 首帧加载时间 |
| pwc卡顿数 | mlp_stall_count.pt | 播放过程卡顿次数 |
| pwc卡顿率 | mlp_stall_rate.pt | 播放过程卡顿比率 |

三个目标分别训练独立的 MLP 模型。

## 项目结构

```
├── dataset.py       # 数据集 + 标准化（train/val/test 切分，scaler fit on train only）
├── model.py         # MLP 模型（Linear → BatchNorm → ReLU → Dropout）
├── train.py         # 训练流程（Adam + ReduceLROnPlateau + Early Stopping）
├── eval.py          # 评估 + 新数据预测
├── util.py          # 工具函数
└── requirements.txt
```

## 模型结构

两层隐藏层的 MLP：`input → 128 → 64 → 1`

每层：Linear → BatchNorm1d → ReLU → Dropout(0.2)

## 训练配置

- 优化器: Adam (lr=1e-3, weight_decay=1e-4)
- 学习率调度: ReduceLROnPlateau (patience=5, factor=0.5)
- Early Stopping: patience=15
- 批大小: 64
- 最大轮数: 100

## 数据切分

按顺序切分（非随机），比例：train 68% / val 12% / test 20%

特征和目标均做 StandardScaler 标准化，仅在训练集上 fit。

## 使用方法

### 训练

```bash
# 训练全部三个目标
python3 train.py --data data.csv

# 训练单个目标
python3 train.py --data data.csv --target ttfb

# 指定 checkpoint 保存目录
python3 train.py --data data.csv --save-dir ./checkpoints
```

### 评估

```bash
python3 eval.py --data data.csv --model checkpoints/mlp_ttfb.pt --target ttfb --mode eval
```

输出指标：MAE、RMSE、R2、MAPE

### 预测新数据

```bash
python3 eval.py --data new_data.csv --model checkpoints/mlp_ttfb.pt --target ttfb --mode predict
```

## Checkpoint 内容

每个 .pt 文件保存：
- model_state_dict: 模型权重
- optimizer_state_dict: 优化器状态
- scaler_X / scaler_y: 标准化器（部署时直接加载，无需重新 fit）
- config: 训练超参数
- epoch / best_val_loss: 训练信息

## 依赖

```bash
pip install -r requirements.txt
```
