import math, copy, pathlib
import xml.etree.ElementTree as ET

# ---------- 你已有的工具函数 ----------
def dist(p, q, use_3d=True):
    return math.dist(p, q) if use_3d else math.hypot(p[0]-q[0], p[1]-q[1])

def route_length(route, use_3d=True):
    pts = [(float(p.attrib['x']), float(p.attrib['y']), float(p.attrib['z']))
           for p in route.findall(".//position")]
    return sum(dist(pts[i], pts[i+1], use_3d) for i in range(len(pts)-1))

def top_long_routes(root, top_k=20, use_3d=True):
    lengths = [ (route_length(r, use_3d), r) for r in root.findall(".//route") ]
    return sorted(lengths, key=lambda x: x[0], reverse=True)[:top_k]

# ---------- 新增：写出前 20 条路线 ----------
def save_top_routes(src_xml, dst_xml, top_k=20, use_3d=True):
    root = ET.parse(src_xml).getroot()
    top_routes = top_long_routes(root, top_k, use_3d)

    new_root = ET.Element("routes")
    for _, r in top_routes:
        # deepcopy 防止修改源树
        new_root.append(copy.deepcopy(r))

    pathlib.Path(dst_xml).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(new_root).write(dst_xml, encoding="utf-8", xml_declaration=True)
    print(f"[OK] 已保存 {len(top_routes)} 条路线到 {dst_xml}")

# ---------- 新增：按长度阈值过滤并保存 ----------
def save_routes_longer_than(src_xml,
                            dst_xml,
                            min_length=100.0,
                            use_3d=True):
    """
    读取 `src_xml`，把所有长度 > min_length (米) 的 <route> 复制到 `dst_xml`。

    依赖现有的 `route_length()`，其余工具函数保持原样。
    """
    root = ET.parse(src_xml).getroot()

    long_routes = [
        r for r in root.findall(".//route")
        if route_length(r, use_3d) > min_length
    ]

    new_root = ET.Element("routes")
    for r in long_routes:
        new_root.append(copy.deepcopy(r))        # 防止修改原树

    pathlib.Path(dst_xml).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(new_root).write(
        dst_xml, encoding="utf-8", xml_declaration=True
    )

    print(f"[OK] 已保存 {len(long_routes)} 条路线 (> {min_length} m) 到 {dst_xml}")

# ---------- 调用 ----------
full_file = "/home/featurize/Bench2Drive/leaderboard/data/bench2drive220.xml"
src_file = "/home/featurize/Bench2Drive/leaderboard/jcz_data/0_merged_routes.xml"
root = ET.parse(src_file).getroot()
save_path = "/home/featurize/Bench2Drive/leaderboard/jcz_data/0_routes_gt100m.xml"
dst_file = "/home/featurize/Bench2Drive/leaderboard/jcz_data/length_top60_routes.xml"   # 想放哪里改这里
save_top_routes(full_file, dst_file, top_k=60, use_3d=True)
# top_routes = top_long_routes(root, top_k=60, use_3d=True)


# save_routes_longer_than(src_file, save_path, min_length=100.0, use_3d=True)