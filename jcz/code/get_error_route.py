import xml.etree.ElementTree as ET

# 你的原始路由文件
input_file = '/home/featurize/Bench2Drive/leaderboard/data/1_jcz_bench2drive220_uniad_speedlimit.xml'
# 输出的失败路由文件
output_file = '/home/featurize/Bench2Drive/leaderboard/jcz_data/0_failed_routes.xml'

# 要提取的失败 route id
failed_ids = {'23659', '23771', '24092', '24258', '25300', '26394', '26401', '26408', '26990', '3048', '3785', '3800', '3936'}

# 读取原始 XML 文件
tree = ET.parse(input_file)
root = tree.getroot()

# 创建新的根节点（保持原始结构）
new_root = ET.Element(root.tag)

# 筛选失败的 route 节点
for route in root.findall('route'):
    if route.get('id') in failed_ids:
        new_root.append(route)

# 写入新的 XML 文件
tree_out = ET.ElementTree(new_root)
tree_out.write(output_file, encoding='utf-8', xml_declaration=True)

print(f"提取完成，共保存 {len(new_root)} 条失败的 routes 到 {output_file}")