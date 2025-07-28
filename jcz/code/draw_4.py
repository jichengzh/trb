import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ========= 0. 全局学术风格 =========
mpl.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": "DejaVu Sans",
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.markersize": 6,
    "axes.unicode_minus": False,
})

# ========= 1. 数据 =========
fps_base = [0.8, 1, 1.25, 1.67, 2, 3, 5, 8, 10, 20, 30]
eff_base = [159.273, 160.354, 161.555, 156.475, 149.319,
            151.795, 150.103, 155.313, 153.227, 157.632, 154.8]
comf_base = [0.436, 0.421, 0.400, 0.418, 0.406,
             0.423, 0.421, 0.380, 0.375, 0.283, 0.272]

# 实时模型
models = ["UniAD", "VAD", "TCP"]
fps_rt  = [2.31, 6.53, 20]
eff_rt  = [142.92, 166.262, 60.34]
comf_rt = [0.424, 0.544, 0.471]

colors_rt  = ["#d62728", "#ff7f0e", "#2ca02c"]   # 红/橙/绿
markers_rt = ["o", "s", "^"]                     # 圆/方/三角

# ========= 2. 横轴索引 =========
x_base = list(range(len(fps_base)))
fps_labels = [str(f) for f in fps_base]
x_rt = [min(range(len(fps_base)), key=lambda i: abs(fps_base[i]-fr))
        for fr in fps_rt]

# ========= 3. 绘图 =========
fig, ax_left = plt.subplots(figsize=(6.5, 4))

# ---- 左轴：Efficiency ----
ax_left.plot(x_base, eff_base,
             color="#1f77b4", lw=1.5, marker='o', label="Efficiency‑Base")
for xr, y, col, mkr in zip(x_rt, eff_rt, colors_rt, markers_rt):
    ax_left.scatter(xr, y, color="#20AAAA", marker=mkr, zorder=5,
                    label=f"Efficiency‑{models[colors_rt.index(col)]}")
ax_left.set_ylabel("Efficiency", color="#1f77b4")
ax_left.tick_params(axis='y', labelcolor="#1f77b4")
ax_left.grid(axis='y', ls='--', lw=0.5)

# ---- 右轴：Comfortness ----
ax_right = ax_left.twinx()
ax_right.plot(x_base, comf_base,
              color="#9467bd", lw=1.5, ls='--', marker='d',
              label="Comfortness‑Base")
for xr, y, col, mkr in zip(x_rt, comf_rt, colors_rt, markers_rt):
    ax_right.scatter(xr, y, color="#915A8E", marker=mkr, 
                     zorder=5, label=f"Comfortness‑{models[colors_rt.index(col)]}")
    
# === 手动扩展纵轴区间 ===
c_min, c_max = min(comf_base + comf_rt), max(comf_base + comf_rt)
margin = 0.1  # 上下各留 0.05
ax_right.set_ylim(c_min - margin, c_max + margin)
ax_right.set_ylabel("Comfortness", color="#9467bd")
ax_right.tick_params(axis='y', labelcolor="#9467bd")



# ---- 公共 x 轴 ----
ax_left.set_xlabel("FPS")
ax_left.set_xticks(x_base)
ax_left.set_xticklabels(fps_labels, rotation=45)

# ---- 合并图例 ----
lines  = ax_left.get_lines() + ax_right.get_lines()
labels = [l.get_label() for l in lines]

ax_left.legend(lines, labels,
               frameon=False,
               ncol=3,                      # 一行 3 列
               loc="lower center",          # 以图例下边缘为参考点
               bbox_to_anchor=(0.5, -0)  # x=0.5 居中，y=-0.25 向下移
)
fig.subplots_adjust(bottom=0.22)
plt.title("Impact of Real‑time Performance on Efficiency & Comfortness")
plt.tight_layout()

# ========= 4. 保存 =========
out_dir = "/home/featurize/output_fig"
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "efficiency_comfortness_twinx.png")
plt.savefig(save_path, dpi=300)
plt.close(fig)
print("✅ 已保存:", save_path)

#两个子图
# ========= 1. 数据 =========
fps_base = [0.8, 1, 1.25, 1.67, 2, 3, 5, 8, 10, 20, 30]
efficiency_base = [159.273, 160.354, 161.555, 156.475, 149.319,
                   151.795, 150.103, 155.313, 153.227, 157.632, 154.8]
comfortness_base = [0.436, 0.421, 0.400, 0.418, 0.406,
                    0.423, 0.421, 0.380, 0.375, 0.283, 0.330]

models        = ["UniAD", "VAD", "TCP"]
fps_rt        = [2.31, 6.53, 20]
efficiency_rt = [142.92, 166.262, 60.34]
comfort_rt    = [0.424, 0.544, 0.471]

colors_rt  = ["#d62728", "#ff7f0e", "#2ca02c"]   # 红/橙/绿
markers_rt = ["o", "s", "^"]

# ========= 2. 横轴索引 (等距) =========
x_base = list(range(len(fps_base)))
fps_labels = [str(f) for f in fps_base]
x_rt = [min(range(len(fps_base)), key=lambda i: abs(fps_base[i]-fr))
        for fr in fps_rt]

# ========= 3. 绘制 Efficiency & Comfortness 子图 =========
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"hspace": 0.18}
)

# ---- Efficiency (上) ----
ax1.plot(x_base, efficiency_base,
         color="#1f77b4", lw=1.5, marker='o', label="Base UniAD")
for xr, y, col, mkr, lbl in zip(x_rt, efficiency_rt, colors_rt, markers_rt, models):
    ax1.scatter(xr, y, color=col, marker=mkr, zorder=5, label=lbl)
ax1.set_ylabel("Efficiency")
ax1.set_title("Impact of Real‑time Performance on Efficiency & Comfortness")
ax1.grid(axis='y', ls='--', lw=0.5)
ax1.legend(frameon=False, ncol=4, loc="upper right")

# ---- Comfortness (下) ----
ax2.plot(x_base, comfortness_base,
         color="#1f77b4", lw=1.5, marker='o', label="Base UniAD")
for xr, y, col, mkr in zip(x_rt, comfort_rt, colors_rt, markers_rt):
    ax2.scatter(xr, y, color=col, marker=mkr, zorder=5)
ax2.set_ylabel("Comfortness")
ax2.set_xlabel("FPS")
ax2.grid(axis='y', ls='--', lw=0.5)

# 统一 x 轴刻度
ax2.set_xticks(x_base)
ax2.set_xticklabels(fps_labels, rotation=45)

# ========= 4. 保存 =========
out_dir = "/home/featurize/output_fig"
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "efficiency_comfortness_subplots.png")
fig.tight_layout()
fig.savefig(save_path, dpi=300)
plt.close(fig)
print("✅ 已保存:", save_path)