#!/usr/bin/env python3
"""
统计 Bench2Drive 评测结果中每个场景的平均 speed 与 desired_speed

使用方法：
    python avg_speed.py \
        --root /home/featurize/Bench2Drive/eval_bench2drive220_uniad_speed_lim0_traj1
脚本会在终端打印结果，也可以通过  --csv 选项把结果保存为 CSV 文件。
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Dict
from typing import Optional


def collect_speeds(meta_dir: Path) -> Tuple[List[float], List[float]]:
    """
    读取 meta_dir 下所有 *.json，返回 speed 列表 和 desired_speed 列表
    """
    speeds, desired_speeds = [], []

    for jf in meta_dir.glob("*.json"):
        try:
            with jf.open("r") as f:
                data = json.load(f)
            # speed / desired_speed 可能写在 root，也可能写在 nested dict，请按需调整
            speeds.append(float(data["speed"]))
            desired_speeds.append(float(data["desired_speed"]))
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[WARN] 跳过文件 {jf}: {e}", file=sys.stderr)
            continue

    return speeds, desired_speeds


def average(values: List[float]) -> float:
    return mean(values) if values else float("nan")


def main(root: Path, csv_out: Optional[Path] = None):
    print(1)
    if not root.exists():
        sys.exit(f"[ERR] 目录不存在: {root}")

    results: Dict[str, Tuple[float, float]] = {}

    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        print(1)
        meta_dir = scenario_dir / "meta"
        if not meta_dir.is_dir():
            print(f"[WARN] 场景 {scenario_dir.name} 缺少 meta 目录，跳过", file=sys.stderr)
            continue

        speeds, desired_speeds = collect_speeds(meta_dir)
        avg_speed = average(speeds)
        avg_desired = average(desired_speeds)
        results[scenario_dir.name] = (avg_speed, avg_desired)

        print(f"{scenario_dir.name:<60} "
              f"avg_speed = {avg_speed:7.3f}  |  avg_desired_speed = {avg_desired:7.3f}")

    # 可选：保存为 CSV
    if csv_out:
        import csv
        with csv_out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "avg_speed", "avg_desired_speed"])
            for name, (s, d) in results.items():
                writer.writerow([name, f"{s:.6f}", f"{d:.6f}"])
        print(f"\nCSV 结果已保存到 {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path,
                        help="包含 39 个 RouteScenario_* 子目录的根目录")
    parser.add_argument("--csv", dest="csv_out", type=Path,
                        help="可选：输出 CSV 文件路径")
    args = parser.parse_args()
    main(args.root, args.csv_out)
