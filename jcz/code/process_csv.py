import pandas as pd
import os

# 读入你的 CSV / TXT 列表
df = pd.read_excel("/home/featurize/jcz/log/fps_samples_2000.csv.xlsx")   # 单列名假设为 'fps'

mask = df['fps'].between(2.5, 2.8)
idx_to_drop = df[mask].index[:100]       # 前 100 条
df_clean = df.drop(idx_to_drop)

save_dir = "/home/featurize/jcz/log"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "fps_clean.csv")
df_clean.to_csv(save_path, index=False)
print("✅ 已保存:", save_path)