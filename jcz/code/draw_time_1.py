import matplotlib.pyplot as plt
import os

# ==== 基础数据 ====
fps_base = [0.8, 1, 1.25, 1.67, 2, 3, 5, 8, 10, 20, 30]
score_base = [35.737, 33.737, 34.049, 35.335, 37.125, 40.242, 39.371, 43.223, 45.643, 48.095, 44.454]

# ==== 三个实时模型 ====
fps_realtime = [2.31, 6.53, 20]
score_realtime = [34.104, 39.922, 42.631]
labels_realtime = ["UniAD", "VAD", "TCP"]
colors_realtime = ["red", "green", "purple"]

# ==== 生成统一的横轴索引 ====
# 先合并所有 FPS 标签（base + realtime）并去重，保证它们在横轴上都有位置
all_fps_labels = [str(f) for f in fps_base]  # 只显示 base 作为主刻度
x_base = list(range(len(fps_base)))  # base 点索引位置

# 为 realtime 单独映射索引位置（在 base fps 中找最接近的插入位置）
x_realtime = []
for fr in fps_realtime:
    # 如果在 base fps 里存在就取对应索引，否则放在末尾（或插值）
    if fr in fps_base:
        x_realtime.append(fps_base.index(fr))
    else:
        # 插入到最接近 fps 的位置
        idx = min(range(len(fps_base)), key=lambda i: abs(fps_base[i]-fr))
        x_realtime.append(idx)

# ==== 绘制 ====
plt.figure(figsize=(9, 5))

# 绘制 Base UniAD 蓝色实线 + 实心圆点（用均匀索引）
plt.plot(x_base, score_base, color='blue', marker='o', linestyle='-', label='Base UniAD')

# 绘制实时模型（用对应索引位置的点）
for xr, s, label, c in zip(x_realtime, score_realtime, labels_realtime, colors_realtime):
    plt.scatter(xr, s, color=c, s=80, label=label)

# 设置横轴为自定义标签，均匀分布
plt.xticks(ticks=x_base, labels=all_fps_labels, rotation=45)

plt.xlabel("FPS (均匀分布标签)")
plt.ylabel("Driving Score")
plt.title("Driving Score vs FPS (均匀分布标签)")
plt.legend()

plt.tight_layout()

# ==== 保存到固定目录 ====
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "driving_score_vs_fps.png")
plt.savefig(save_path, dpi=150)

plt.show()
print(f"✅ 图像已保存到: {save_path}")
