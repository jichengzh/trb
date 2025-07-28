import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from brokenaxes import brokenaxes

# ========== 0. 全局学术风格 ==========
mpl.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": "DejaVu Sans",
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.markersize": 6,
    "axes.unicode_minus": False,     # 负号正常显示
})

# === 新的实验模型 ===
models_extra = [
    "baseline",
    "(1) occ",
    "(2) occ_seg",
    "(3) occ_seg_motion",
    "(4) motion_seg",
    "(5) occ_motion",
    "(6) seg",
    "(7) motion"
]

fps_extra = [2.31, 2.59, 3.27, 3.58, 2.89, 2.54, 2.72, 2.43]

driving_score_extra = [34.104, 39.886, 38.151, 18.67, 29.868, 19.161, 35.831, 32.01]
efficiency_extra = [142.92, 157.36, 158.87, 136.92, 146.75, 134.10, 150.43, 147.09]
comfortness_extra = [0.437, 0.406, 0.418, 0.646, 0.538, 0.645, 0.427, 0.619]
energy_extra = [191.29, 168.56, 152.49, 138.74, 161.47, 154.82, 175.22, 177.54]

# ========== 1. 数据 ==========
# fps_base = [0.8, 1, 1.25, 1.67, 2, 3, 5, 8, 10, 20, 30]
fps_base = [1.67, 2, 3, 5]
# driving_score_base = [35.737, 33.737, 34.049, 35.335, 37.125,
#                       40.242, 39.371, 43.223, 45.643, 48.095, 44.454]
driving_score_base = [35.335, 37.125, 40.242, 39.371]
efficiency_base = [159.273, 160.354, 161.555, 156.475, 149.319,
                   151.795, 150.103, 155.313, 153.227, 157.632, 154.8]
comfortness_base = [0.436, 0.421, 0.4, 0.418, 0.406,
                    0.423, 0.421, 0.38, 0.375, 0.283, 0.33]

# 三个实时模型
models          = ["UniAD", "VAD", "TCP"]
fps_realtime    = [2.31, 6.53, 20]
driving_score_rt = [34.104, 39.922, 42.631]
efficiency_rt    = [142.92, 166.262, 60.34]
comfortness_rt   = [0.424, 0.544, 0.471]

# 学术调色 & marker
colors_rt  = ["#d62728", "#ff7f0e", "#2ca02c"]   # 蓝 / 橙 / 绿
markers_rt = ["o", "s", "^"]

# ========== 2. 横轴索引 (等距) ==========
x_base = list(range(len(fps_base)))
fps_labels = [str(f) for f in fps_base]
# 把实时模型 FPS 映射到最近的横轴索引
x_rt = [min(range(len(fps_base)), key=lambda i: abs(fps_base[i]-fr))
        for fr in fps_realtime]
# x_base = list(range(len(time_cost_base)))
# fps_labels = [str(f) for f in time_cost_base]
# # 把实时模型 FPS 映射到最近的横轴索引
# x_rt = [min(range(len(time_cost_base)), key=lambda i: abs(time_cost_base[i]-fr))
#         for fr in fps_realtime]

# ========== 3. 绘图函数 ==========
def plot_metric(base_values, rt_values, ylabel, filename,
                extra_fps=None, extra_values=None, extra_labels=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    # 基准折线
    ax.plot(x_base, base_values,
            color="#1f77b4", linewidth=1.5,
            marker='o', label="Base UniAD")

    # # 实时模型点
    # for xr, y, lbl, col, mkr in zip(x_rt, rt_values, models, colors_rt, markers_rt):
    #     ax.scatter(xr, y, color=col, marker=mkr, zorder=5, label=lbl)
    
    
    # ✅ 新增: 额外实验散点（映射到最接近 fps_base 的 x）
    # if extra_fps and extra_values:
    #     x_extra = [
    #         min(range(len(fps_base)), key=lambda i: abs(fps_base[i] - fr))
    #         for fr in extra_fps
    #     ]
    #     for xe, val, lbl in zip(x_extra, extra_values, extra_labels):
    #         ax.scatter(xe, val, color="#9467bd", marker="D", s=40,
    #                    zorder=5, label=lbl)  # 紫色菱形

    # ==== 关键改动：为 Driving Score 单独设置 y‑axis 下限 ====
    if ylabel.lower().startswith("driving"):
        ax.set_ylim(28, 41)     # 下限 33，自动计算上限
    # 轴与网格
    ax.set_xticks(x_base)
    ax.set_xticklabels(fps_labels, rotation=45)
    ax.set_xlabel("FPS")
    ax.set_ylabel(ylabel)
    ax.set_title(f"The impact of real‑time performance on {ylabel}")
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color="#d0d0d0")

    ax.legend(frameon=False)
    fig.tight_layout()

    # 保存
    out_dir = "/home/featurize/output_fig"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ 已保存: {save_path}")
# ================== 断轴 ========================================================================


# ========== 4. 生成三张图 ==========


# === 把额外实验的 driving score 绘制进去 ===
plot_metric(
    driving_score_base, driving_score_rt,
    ylabel="Driving Score", filename="driving_score_vs_fps_with_extra",
    extra_fps=fps_extra,
    extra_values=driving_score_extra,
    extra_labels=models_extra
)
# plot_metric_brokenaxis(
#     driving_score_base,
#     ylabel="Driving Score", filename="driving_score_vs_fps_with_extra",
#     extra_fps=fps_extra,
#     extra_values=driving_score_extra,
#     extra_labels=models_extra
# )
# plot_metric(efficiency_base, efficiency_rt,
#             ylabel="Efficiency",    filename="efficiency_vs_fps")
# plot_metric(comfortness_base, comfortness_rt,
#             ylabel="Comfortness",  filename="comfortness_vs_fps")