#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计评测结果 JSON 中的三类碰撞：
  · collisions_layout
  · collisions_pedestrian
  · collisions_vehicle

用法:
    python count_all_collisions.py eval_results.json
"""

import sys, json, pathlib
from collections import Counter, defaultdict

def load_records(json_path: str):
    """兼容 ① {"_checkpoint": {"records":[...]}} ② {"records":[...]} ③ 直接是 list"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):                 # ③
        return data
    if "records" in data:                     # ②
        return data["records"]
    if "_checkpoint" in data:                 # ①
        return data["_checkpoint"]["records"]

    raise ValueError("未找到 'records' 列表，请检查 JSON 结构")

def count_collisions(records):
    """返回总体 Counter 以及按 route_id 分组的明细"""
    total = Counter(layout=0, pedestrian=0, vehicle=0)
    per_route = defaultdict(Counter)

    for rec in records:
        rid   = rec.get("route_id", "<no_id>")
        infra = rec.get("infractions", {})

        # 三类碰撞分别计数
        layout_n     = len(infra.get("collisions_layout", []))
        ped_n        = len(infra.get("collisions_pedestrian", []))
        vehicle_n    = len(infra.get("collisions_vehicle", []))

        total["layout"]     += layout_n
        total["pedestrian"] += ped_n
        total["vehicle"]    += vehicle_n

        per_route[rid]["layout"]     += layout_n
        per_route[rid]["pedestrian"] += ped_n
        per_route[rid]["vehicle"]    += vehicle_n

    return total, per_route

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python count_all_collisions.py <eval_results.json>")
        sys.exit(1)

    json_file = pathlib.Path(sys.argv[1])
    if not json_file.exists():
        print(f"文件不存在: {json_file}")
        sys.exit(1)

    records = load_records(json_file)
    total, per_route = count_collisions(records)

    print("\n==== 全文件碰撞统计 ====")
    print(f"  • layout     : {total['layout']}")
    print(f"  • pedestrian : {total['pedestrian']}")
    print(f"  • vehicle    : {total['vehicle']}")
    print(f"  • 场景总数     : {len(records)}")

    # print("\n---- 各 route 明细 ----")
    # for rid, cnt in per_route.items():
    #     print(f"{rid:>20s}  |  layout={cnt['layout']:2d}  "
    #           f"pedestrian={cnt['pedestrian']:2d}  vehicle={cnt['vehicle']:2d}")