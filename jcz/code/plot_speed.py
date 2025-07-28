#!/usr/bin/env python3
"""
plot_speeds_align.py
把 /home/featurize/output/result 下所有 csv 合并成同一个表，
让三条（或更多）曲线共享同一组 scenario 作为 x 轴。
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_all(root: Path):
    """读取目录下所有 csv，并合并成
       index = scenario
       columns = {<csv_name>_speed, <csv_name>_desire}
    """
    dfs = []
    names = []  # 记住文件名顺序，方便画图时配色/图例一致
    for csv in sorted(root.glob("*.csv")):
        df = pd.read_csv(csv, usecols=["scenario", "avg_speed", "avg_desired_speed"])
        # 让 scenario 成为行索引，其他列改名以避免重名
        tag = csv.stem            # 例如 speedlim+2
        df = df.set_index("scenario").rename(
            columns={
                "avg_speed":        f"{tag}_speed",
                "avg_desired_speed": f"{tag}_desire",
            }
        )
        dfs.append(df)
        names.append(tag)

    # 外连接：遇到缺少的 scenario 用 NaN 填
    merged = pd.concat(dfs, axis=1, sort=True)
    merged = merged.sort_index()  # scenario 排一下序，横轴更稳
    return merged, names


def plot_block(df, cols, ylabel, out_png):
    fig, ax = plt.subplots(figsize=(14, 6))  # 推荐使用对象式 API
    x = range(len(df.index))  # 统一的 x 轴位置

    for col in cols:
        ax.plot(x, df[col], marker="o", label=col.replace("_speed", "")
                                              .replace("_desire", ""))

    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=90, fontsize=6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)  # 🟢 添加这行确保每张图清理
    print(f"✅  {out_png} saved")


def main(root: Path, show: bool):
    merged, names = load_all(root)

    # ① 速度
    speed_cols   = [f"{n}_speed"   for n in names]
    plot_block(merged, speed_cols,
               ylabel="Average speed (m/s)",
               out_png=root / "avg_speed_all.png")

    # ② desired speed
    desire_cols  = [f"{n}_desire"  for n in names]
    plot_block(merged, desire_cols,
               ylabel="Average desired speed (m/s)",
               out_png=root / "avg_desired_speed_all.png")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("/home/featurize/output/result"),
                        help="目录，里面放的是多个 csv")
    parser.add_argument("--show", action="store_true",
                        help="画完之后弹窗显示")
    args = parser.parse_args()
    main(args.root, args.show)