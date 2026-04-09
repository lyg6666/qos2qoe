"""
QoS和QoE随时间变化的折线图
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
# client
df = pd.read_csv('client_metrics_0327_2000_to_0328_2000.csv', encoding='utf-8')
# server
# df = pd.read_csv('server_metrics_0327_2000_to_0328_2000.csv', encoding='utf-8')

df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d %H:%M')
df.set_index('时间', inplace=True)
output_dir = 'client_metrics_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        continue
        
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df[col], color='#1f77b4', linewidth=1.5)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(f'{col} 随时间变化趋势', fontsize=15)
    ax.set_xlabel('时间 (2026-03-27)', fontsize=12)
    ax.set_ylabel(col, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    safe_col_name = str(col).replace('/', '_').replace('\\', '_')
    plt.savefig(f'{output_dir}/{safe_col_name}.png', dpi=150)
    plt.close()

print(f"所有图片已保存在 '{output_dir}' 文件夹中。")