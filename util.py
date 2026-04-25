# 工具函数
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import (
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_DATASET_OUTPUT_DIR,
    DEFAULT_EVAL_PLOT_DIR,
    DATA_SPLIT_CONFIG,
    MERGE_CONFIG,
    VIS_CONFIG,
)


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
    """
    数据预处理：
    读取多个 client/server 数据文件，分别拼接后按共同时间点过滤。
    仅保留 client/server 双方都存在的时间点。
    数据中有一些行是重复的，处理过程中也会删除重复的数据行。
    同时这个数据存在一些时间点有多行数据的情况，保留非空字段最多的一行。

    参数:
        rawdataset_dir: 原始数据目录。
        output_dir: 输出目录。默认输出到当前目录外的 ../dataset。
        time_col: 时间列名，默认 "时间"。

    返回:
        client_filtered, server_filtered, merged_filtered
    """
    raw_dir = Path(rawdataset_dir) # rawdata 目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_suffixes = set(MERGE_CONFIG["supported_suffixes"])
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_suffixes]

    client_files = sorted([p for p in files if "client" in p.stem.lower()])
    server_files = sorted([p for p in files if "server" in p.stem.lower()])

    def _read_table(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_excel(path)

    def _dedupe_by_time_keep_most_complete(df: pd.DataFrame) -> pd.DataFrame:
        # 同一时间点可能存在多行，保留非空字段最多的一行
        temp = df.copy()
        value_cols = [c for c in temp.columns if c != time_col]
        temp["__filled_count"] = temp[value_cols].notna().sum(axis=1)
        keep_idx = temp.groupby(time_col, sort=False)["__filled_count"].idxmax()
        deduped = temp.loc[keep_idx].drop(columns=["__filled_count"])
        return deduped.sort_values(time_col).reset_index(drop=True)

    def _concat_and_sort(file_list, role_name):
        dfs = []
        for fp in file_list:
            df = _read_table(fp)
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col])
            dfs.append(df)

        merged_role_df = pd.concat(dfs, ignore_index=True)
        merged_role_df = _dedupe_by_time_keep_most_complete(merged_role_df)
        return merged_role_df

    client_all = _concat_and_sort(client_files, "client")
    server_all = _concat_and_sort(server_files, "server")

    common_times = pd.Index(client_all[time_col]).intersection(pd.Index(server_all[time_col]))

    client_filtered = (
        client_all[client_all[time_col].isin(common_times)]
        .pipe(_dedupe_by_time_keep_most_complete)
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    server_filtered = (
        server_all[server_all[time_col].isin(common_times)]
        .pipe(_dedupe_by_time_keep_most_complete)
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    output_paths = {
        "client": output_dir / "client.csv",
        "server": output_dir / "server.csv",
        "merged": output_dir / "dataset.csv",
    }

    target_cols = MERGE_CONFIG["target_cols"]
    client_targets = client_filtered[[time_col] + target_cols].copy()
    merged_filtered = pd.concat(
        [
            client_targets.reset_index(drop=True),
            server_filtered.drop(columns=[time_col], errors="ignore").reset_index(drop=True),
        ],
        axis=1,
    )

    client_filtered.to_csv(output_paths["client"], index=False, encoding="utf-8-sig")
    server_filtered.to_csv(output_paths["server"], index=False, encoding="utf-8-sig")
    merged_filtered.to_csv(output_paths["merged"], index=False, encoding="utf-8-sig")

    return client_filtered, server_filtered, merged_filtered


def visualize_eval_results(
    y_true,
    y_pred,
    log_count,
    target_name,
    save_path=None,
    max_points=VIS_CONFIG["max_points"],
):
    """
    可视化测试集标签、预测值、日志数，并保存图片。

    参数:
        y_true: 真实标签
        y_pred: 预测值
        log_count: 日志数
        target_name: 目标名（ttfb什么的）
        save_path: 图片保存路径，默认 ../output/eval/{target_name}_pred_vs_true.png
        max_points: 折线图显示点数上限；None 表示显示全部
    """
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

    # 左轴: 真实值与预测值
    line_true = ax_left.plot(idx, y_true[:show_n], label="True", linewidth=1.6, color="#1f77b4")
    line_pred = ax_left.plot(idx, y_pred[:show_n], label="Pred", linewidth=1.2, color="#ff7f0e")
    ax_left.set_xlabel("Sample Index")
    ax_left.set_ylabel(f"{target_name} Value")
    ax_left.grid(alpha=0.25)

    # 右轴: 日志数
    line_log = ax_right.plot(idx, log_count[:show_n], label="Log Count", linewidth=1.0, color="#2ca02c", alpha=0.75)
    ax_right.set_ylabel("log count")

    all_lines = line_true + line_pred + line_log
    all_labels = [l.get_label() for l in all_lines]
    ax_left.legend(all_lines, all_labels, loc="upper right")
    ax_left.set_title(f"{target_name}: True/Pred + Log Count (First {show_n})")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def build_test_set_with_checkpoint_scalers(
    df,
    feature_cols,
    target_col,
    scaler_X,
    scaler_y,
    log_col=VIS_CONFIG["log_col"],
    val_ratio=DATA_SPLIT_CONFIG["val_ratio"],
    test_ratio=DATA_SPLIT_CONFIG["test_ratio"],
):
    """按训练时同样切分规则构造测试集，并仅使用 checkpoint 中的 scaler 做变换。"""
    from dataset import PredictDataset

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    test_idx = int(len(X) * (1 - test_ratio))
    val_idx = int(test_idx * (1 - val_ratio))
    X_test = X[test_idx:]
    y_test = y[test_idx:]
    log_test = df[log_col].values.astype(np.float32)[test_idx:]

    X_test = np.nan_to_num(scaler_X.transform(X_test), nan=0.0).astype(np.float32)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)
    return PredictDataset(X_test, y_test), log_test


def explain_qoe_change_with_shap(
    model,
    df,
    feature_cols,
    scaler_X,
    scaler_y,
    target_name,
    sample_index=-1,
    time_col=MERGE_CONFIG["time_col"],
    window_minutes=10,
    top_k=3,
    background_size=128,
):
    """使用 SHAP 解释单样本 QoE 预测相对前10分钟基线的上升/下降原因。"""
    import shap

    if sample_index < 0:
        sample_index = len(df) + sample_index
    if sample_index < 0 or sample_index >= len(df):
        raise IndexError(f"sample_index 超出范围: {sample_index}, 数据长度={len(df)}")

    if time_col not in df.columns:
        raise ValueError(f"数据缺少时间列: {time_col}")

    work_df = df.copy()
    work_df[time_col] = pd.to_datetime(work_df[time_col], errors="coerce")
    work_df = work_df.dropna(subset=[time_col]).reset_index(drop=True)

    if sample_index >= len(work_df):
        raise IndexError(f"时间清洗后 sample_index 超出范围: {sample_index}, 数据长度={len(work_df)}")

    x_raw_all = work_df[feature_cols].values.astype(np.float32)
    x_scaled_all = np.nan_to_num(scaler_X.transform(x_raw_all), nan=0.0).astype(np.float32)

    ts_current = work_df.loc[sample_index, time_col]
    ts_start = ts_current - pd.Timedelta(minutes=window_minutes)
    baseline_mask = (work_df[time_col] >= ts_start) & (work_df[time_col] < ts_current)
    baseline_idx = np.where(baseline_mask.values)[0]
    if len(baseline_idx) == 0:
        raise ValueError(f"样本 {sample_index} 前 {window_minutes} 分钟内无可用基线数据")

    baseline_scaled = x_scaled_all[baseline_idx]
    if len(baseline_scaled) > background_size:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(baseline_scaled), size=background_size, replace=False)
        baseline_scaled = baseline_scaled[pick]

    x_current_scaled = x_scaled_all[sample_index:sample_index + 1]

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_current_scaled, dtype=torch.float32)).cpu().numpy().flatten()[0]
        pred_value = scaler_y.inverse_transform(np.array([[pred_scaled]], dtype=np.float32)).flatten()[0]

    with torch.no_grad():
        base_preds_scaled = model(torch.tensor(baseline_scaled, dtype=torch.float32)).cpu().numpy().flatten()
    base_pred_value = scaler_y.inverse_transform(base_preds_scaled.reshape(-1, 1)).flatten().mean()

    def _model_predict_real(x_scaled_np):
        with torch.no_grad():
            y_scaled = model(torch.tensor(x_scaled_np, dtype=torch.float32)).cpu().numpy()
        return scaler_y.inverse_transform(y_scaled).flatten()

    explainer = shap.KernelExplainer(_model_predict_real, baseline_scaled)
    shap_values = explainer.shap_values(x_current_scaled, nsamples=min(200, max(20, len(feature_cols) * 2)))
    shap_values = np.array(shap_values).reshape(-1)

    contrib_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_contribution": shap_values,
        "abs_contribution": np.abs(shap_values),
        "current_value": work_df.loc[sample_index, feature_cols].values,
    }).sort_values("abs_contribution", ascending=False)

    top_df = contrib_df.head(top_k).copy()
    direction = "上升" if (pred_value - base_pred_value) >= 0 else "下降"
    delta = pred_value - base_pred_value

    lines = [
        f"本次预测 {target_name} {pred_value:.4f}，相较前{window_minutes}分钟基线 {base_pred_value:.4f}，变化 {delta:+.4f}（{direction}）。"
    ]
    for _, row in top_df.iterrows():
        lines.append(
            f"{row['feature']} 贡献 {row['shap_contribution']:+.4f}"
        )

    return {
        "sample_index": int(sample_index),
        "timestamp": str(ts_current),
        "prediction": float(pred_value),
        "baseline_prediction": float(base_pred_value),
        "delta": float(delta),
        "direction": direction,
        "top_contributors": top_df[["feature", "shap_contribution", "current_value"]].to_dict(orient="records"),
        "message": "；".join(lines),
    }
