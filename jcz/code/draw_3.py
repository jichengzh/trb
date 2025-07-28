import matplotlib.pyplot as plt

# ==== 1. Base 折线数据 ====
fps_base = [0.8, 1, 1.25, 1.67, 2, 3, 5, 8, 10, 20, 30]
driving_score_base = [35.737, 33.737, 34.049, 35.335, 37.125, 40.242, 39.371, 43.223, 45.643, 48.095, 44.454]

# 横轴等距索引
x_base = list(range(len(fps_base)))
fps_labels = [str(f) for f in fps_base]

# ==== 2. 模型 Driving Score ====
models = ["UniAD", "VAD"]
model_fps_center = [2.31, 6.53]      # 实际FPS
model_score = [34.104, 39.922]     # Driving Score
colors = ["red", "green"]

# ==== 3. 手动误差棒（Q1/Q3） ====
# UniAD: Q1=1.5 Q3=2.2
# VAD: Q1=7 Q3=9
q1_q3 = {
    "UniAD": (1.5, 2.2),
    "VAD": (6, 7)
}

# 误差棒映射到索引坐标
model_x_idx = []
xerr_left_idx = []
xerr_right_idx = []

for model, fps_c in zip(models, model_fps_center):
    # 找到中心fps在fps_base中的索引位置
    center_idx = min(range(len(fps_base)), key=lambda i: abs(fps_base[i] - fps_c))
    q1, q3 = q1_q3[model]
    # 找到q1,q3最接近的索引
    idx_q1 = min(range(len(fps_base)), key=lambda i: abs(fps_base[i] - q1))
    idx_q3 = min(range(len(fps_base)), key=lambda i: abs(fps_base[i] - q3))
    
    model_x_idx.append(center_idx)
    # 左右误差=索引差
    xerr_left_idx.append(center_idx - idx_q1)
    xerr_right_idx.append(idx_q3 - center_idx)

# ==== 4. 绘制 ====
plt.figure(figsize=(9,6))

# Base折线（用索引）
plt.plot(x_base, driving_score_base, color='blue', marker='o', label="Base UniAD")

# 模型误差棒
for xi, yi, l, r, label, c in zip(model_x_idx, model_score, xerr_left_idx, xerr_right_idx, models, colors):
    plt.errorbar(
        xi, yi,
        xerr=[[l],[r]],
        fmt='o', color=c, ecolor=c, elinewidth=2, capsize=6, label=label
    )
    plt.text(xi + 0.2, yi, label, fontsize=10, color=c)

# 设置横轴刻度等距显示
plt.xticks(ticks=x_base, labels=fps_labels, rotation=45)
plt.xlabel("FPS")
plt.ylabel("Driving Score")
plt.title("The impact of real-time performance on Driving Score (with Q1~Q3 FPS variability)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./output_fig/driving_score_q1q3_discrete_fps_axis.png", dpi=150)
plt.show()