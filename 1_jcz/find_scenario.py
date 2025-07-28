import xml.etree.ElementTree as ET

keywords = [
    "SignalizedJunctionLeftTurn",
    "SignalizedJunctionRightTurn",
    "NonSignalizedJunctionLeftTurn",
    "NonSignalizedJunctionRightTurn",
    "EnterActorFlow",
    "EnterActorFlowV2",
    "CrossingBicycleFlow",
    "InterurbanActorFlow",
    "InterurbanAdvancedActorFlow",
    "HighwayExit",
    "MergerIntoSlowTraffic",
    "MergerIntoSlowTrafficV2",
    "HazardAtSideLane",
    "HazardAtSideLaneTwoWays"
]

def extract_and_save_routes(xml_path, keywords, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    new_root = ET.Element('routes')
    count = 0

    for route in root.findall('route'):
        scenarios = route.find('scenarios')
        if scenarios is not None:
            for scenario in scenarios.findall('scenario'):
                scenario_type = scenario.get('type')
                if scenario_type in keywords:
                    new_root.append(route)
                    count += 1
                    break  # 只要有一个 scenario type 匹配即可

    # 保存新xml
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"共找到 {count} 个符合条件的 route，已保存到 {output_path}")

# 过滤速度
def extract_and_save_routes2(xml_path, keywords, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    new_root = ET.Element('routes')
    count = 0

    for route in root.findall('route'):
        scenarios = route.find('scenarios')
        if scenarios is not None:
            for scenario in scenarios.findall('scenario'):
                scenario_type = scenario.get('type')
                if scenario_type in keywords:
                    flow_speed_elem = scenario.find('flow_speed')
                    if flow_speed_elem is None:
                        # 没有flow_speed节点，保留
                        new_root.append(route)
                        count += 1
                        break
                    else:
                        flow_speed_value = flow_speed_elem.get('value')
                        if flow_speed_value is not None and float(flow_speed_value) <= 11:
                            new_root.append(route)
                            count += 1
                            break


    # 保存新xml
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"共找到 {count} 个符合条件的 route，已保存到 {output_path}")
# 用法示例
# xml_path = '/home/featurize/Bench2Drive/leaderboard/data/bench2drive220.xml'         # 原始xml文件路径
xml_path = '/home/featurize/Bench2Drive/leaderboard/data/jcz_bench2drive220_uniad.xml'  # 原始xml文件路径
output_path = '/home/featurize/Bench2Drive/leaderboard/data/jcz_bench2drive220_uniad_speedlimit.xml'  # 新xml文件路径
extract_and_save_routes2(xml_path, keywords, output_path)