import matplotlib.pyplot as plt
import re
import os
import statistics
# ============ 1. 从日志中提取 Computation time 并计算 FPS ============
def parse_fps_from_log(path):
    """
    从log文件提取 'Computation time: xxx s' 并计算 FPS = 1 / time
    """
    with open(path, "r") as f:
        text = f.read()

    # 提取所有 computation time 数值
    comp_times = [float(x) for x in re.findall(r"Computation time:\s*([\d.]+)s", text)]

    # 计算 FPS (避免除0)
    fps_values = [1.0 / t if t > 0 else 0.0 for t in comp_times]
    return fps_values

def filter_outliers(data, method="3sigma"):
    """
    过滤离群点
    method="3sigma" 按正态分布假设用 μ±3σ 过滤
    method="iqr"    用四分位距(IQR)过滤 (Q1-1.5IQR, Q3+1.5IQR)
    """
    if len(data) < 5:
        return data  # 太少就不处理

    if method == "3sigma":
        mu = statistics.mean(data)
        sigma = statistics.stdev(data)
        lower, upper = mu - 3 * sigma, mu + 3 * sigma

    elif method == "iqr":
        q1, q3 = statistics.quantiles(data, n=4)[0], statistics.quantiles(data, n=4)[2]
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    else:
        raise ValueError("method must be '3sigma' or 'iqr'")

    filtered = [x for x in data if lower <= x <= upper]
    print(f"[{method}] 原始 {len(data)} 条 -> 过滤后 {len(filtered)} 条，区间[{lower:.2f}, {upper:.2f}]")
    return filtered


# ============ 2. 三个模型对应日志路径 ============
log_files = {
    "UniAD": "/home/featurize/Bench2Drive/leaderboard/jcz_data/1_jcz_bench2drive220_uniad_speedlimit_3_4_uniad_xiaorong_onlyseg_traj.log",
    "VAD":   "/home/featurize/Bench2Drive/leaderboard/jcz_data/1_jcz_bench2drive220_uniad_speedlimit_2_3_vad_traj.log",
    # "TCP":   "/home/featurize/Bench2Drive/leaderboard/jcz_data/1_jcz_bench2drive220_uniad_speedlimit_2_4_tcp_merge_ctrl_traj.log"
}

# 依次读取每个模型的FPS数据
fps_data = {name: parse_fps_from_log(path) for name, path in log_files.items()}
fps_data["VAD"] = filter_outliers(fps_data["VAD"])
# ============ 3. 绘制箱型图 (横轴是模型名称，纵轴是真实FPS分布) ============
plt.figure(figsize=(8, 5))

# 箱型数据 & 标签
box_data   = [
    fps_data["UniAD"], 
    fps_data["VAD"], 
    # fps_data["TCP"]
    ]
box_labels = ["UniAD", "VAD"]

bp = plt.boxplot(
    box_data,
    labels=box_labels,
    patch_artist=True,
    widths=0.6,
    showmeans=True  # 显示均值点
)

# 上色
colors = ["red", "green", "purple"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.4)

plt.ylabel("FPS")
plt.title("FPS distribution of UniAD / VAD / TCP (from log Computation time)")
plt.grid(axis='y')

# 保存
os.makedirs("./output_fig", exist_ok=True)
save_path = "./output_fig/fps_distribution_boxplot.png"
plt.tight_layout()
plt.savefig(save_path, dpi=150)
plt.show()

print(f"✅ FPS箱型图已保存到: {save_path}")
