"""
qos与qoe所有指标相关性的热力图
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
df_client = pd.read_csv('client_metrics_0327_2000_to_0328_2000.csv', encoding='utf-8')
df_server = pd.read_csv('server_metrics_0327_2000_to_0328_2000.csv', encoding='utf-8')
df_client['时间'] = pd.to_datetime(df_client['时间'])
df_server['时间'] = pd.to_datetime(df_server['时间'])
client_cols = [col for col in df_client.columns if col != '时间' and pd.api.types.is_numeric_dtype(df_client[col])]
server_cols = [col for col in df_server.columns if col != '时间' and pd.api.types.is_numeric_dtype(df_server[col])]

print(f"客户端指标 {len(client_cols)} 个，服务端指标 {len(server_cols)} 个。")
df_merged = pd.merge(df_client, df_server, on='时间', how='inner', suffixes=('_客户端', '_服务端'))
final_client_cols = [c for c in df_merged.columns if c in client_cols or c.endswith('_客户端')]
final_server_cols = [c for c in df_merged.columns if c in server_cols or c.endswith('_服务端')]
# 计算完整相关性矩阵 (使用 spearman)
corr_matrix = df_merged[final_client_cols + final_server_cols].corr(method='spearman')
cross_corr = corr_matrix.loc[final_client_cols, final_server_cols]
corr_pairs = cross_corr.unstack().reset_index()
corr_pairs.columns = ['服务端指标', '客户端指标', '相关系数']
corr_pairs['相关性绝对值'] = corr_pairs['相关系数'].abs()
corr_pairs = corr_pairs.sort_values(by='相关性绝对值', ascending=False).dropna()
output_csv = 'QoE_QoS_Correlation_Rank.csv'
corr_pairs.to_csv(output_csv, index=False, encoding='utf-8-sig')
plt.figure(figsize=(60, 20))
sns.heatmap(
    cross_corr,
    annot=True,        
    cmap='coolwarm',   
    fmt=".2f",      
    linewidths=.5,
    vmin=-1, vmax=1,  
    square=True
)
plt.xlabel('QoS', fontsize=12)
plt.ylabel('QoE', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Correlation_Heatmap.png', dpi=150)