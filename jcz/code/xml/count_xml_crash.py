import json
import sys

def analyze_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 如果是字典并包含"_checkpoint.records"，提取 records
    if isinstance(data, dict) and "_checkpoint" in data:
        routes = data["_checkpoint"].get("records", [])
    elif isinstance(data, list):
        routes = data
    else:
        print("❌ 未知 JSON 结构，无法解析")
        return

    total_routes = len(routes)
    failed_routes = 0
    total_collisions = 0
    layout_count = 0
    ped_count = 0
    veh_count = 0

    for route in routes:
        # 统计 Failed 路由
        if route.get("status", "").startswith("Failed"):
            failed_routes += 1

        # 获取 infractions 下的三类碰撞
        infractions = route.get("infractions", {})
        cl = len(infractions.get("collisions_layout", []))
        cp = len(infractions.get("collisions_pedestrian", []))
        cv = len(infractions.get("collisions_vehicle", []))

        layout_count += cl
        ped_count += cp
        veh_count += cv
        total_collisions += (cl + cp + cv)
    collisions_score = layout_count*(1-0.65)+ped_count*(1-0.5)+veh_count*(1-0.6)
    print(f"📊 总路由数量: {total_routes}")
    print(f"❌ Failed 路由数量: {failed_routes}")
    print(f"🚗 碰撞违规总数: {total_collisions}")
    print(f"    • layout碰撞: {layout_count}")
    print(f"    • 行人碰撞: {ped_count}")
    print(f"    • 车辆碰撞: {veh_count}")
    print(f"事故: {collisions_score}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("✅ 用法示例:")
        print("   python stats_routes.py results.json")
    else:
        analyze_json(sys.argv[1])
