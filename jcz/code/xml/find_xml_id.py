import xml.etree.ElementTree as ET
from pathlib import Path

def collect_route_ids(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # XPath “.//route” 找所有后代 route 元素
    return {route.attrib["id"] for route in root.findall(".//route")}

# file1 = "/home/featurize/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml"
# file2 = "/home/featurize/Bench2Drive/leaderboard/data/1_jcz_bench2drive220_uniad_speedlimit.xml"

# ids_1 = collect_route_ids(file1)
# ids_2 = collect_route_ids(file2)

# print("文件1的 id:", sorted(ids_1))
# print("文件2的 id:", sorted(ids_2))
# print("二者交集 :", sorted(ids_1 & ids_2))

