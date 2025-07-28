import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# === 1. 读取数据 ==========================================================
file_path = Path("/home/featurize/jcz/log/planning.txt")   # 修改为实际路径
with file_path.open("r") as f:
    raw = f.read()

# split → strip → float
values = np.array([float(x) for x in raw.replace("\n", "").split(",") if x.strip()])
values = values[:10000]
# === 2. 绘制折线图 =======================================================
plt.rcParams.update({  # 简单学术风格
    "figure.figsize": (7, 4),
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
})

fig, ax = plt.subplots()
ax.plot(np.arange(len(values)), values, marker="", linestyle="-", color="#1f77b4")
ax.set_xlabel("Frame Index")
ax.set_ylabel("Planning Latency (ms)")
ax.set_title("UniAD Planning Latency per Frame")
ax.set_ylim(bottom=0)   # 从 0 开始，方便观察峰值

# 可选：突出异常峰值 
# threshold = np.percentile(values, 95)      # 95% 位置作为异常阈值
# abnormal_idx = np.where(values > threshold)[0]
# ax.scatter(abnormal_idx, values[abnormal_idx], color="red", s=20, label="Top 5 % Peaks")
# ax.legend(frameon=False)

# === 3. 保存 & 显示 =======================================================
# fig.tight_layout()
# fig.savefig("planning_latency.png", dpi=300, bbox_inches="tight")
# plt.show()

out_dir = "/home/featurize/output_fig"
save_path = os.path.join(out_dir, "efficiency_comfortness_subplots.png")
fig.tight_layout()
fig.savefig(save_path, dpi=300)
plt.close(fig)
print("✅ 已保存:", save_path)