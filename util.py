# 工具函数
import pandas as pd
from pathlib import Path


def read_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    return pd.read_csv(p)
