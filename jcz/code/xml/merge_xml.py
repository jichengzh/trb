import sys, xml.etree.ElementTree as ET, copy, pathlib

def merge_route_xml(src_files, dst_file):
    """把多个 XML 文件中的 <route> 合并到 dst_file，去除 id 重复"""
    new_root = ET.Element("routes")
    seen     = set()                       # 已收集的 route id

    for path in src_files:
        root = ET.parse(path).getroot()
        for r in root.findall(".//route"):
            rid = r.attrib.get("id")
            if rid in seen:
                continue                   # 同 id 已有 -> 跳过
            seen.add(rid)
            new_root.append(copy.deepcopy(r))

    pathlib.Path(dst_file).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(new_root).write(dst_file,
                                   encoding="utf-8",
                                   xml_declaration=True)
    print(f"[OK] 合并完成，共 {len(seen)} 条路线写入 {dst_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python merge_routes.py file1.xml file2.xml file3.xml [output.xml]")
        sys.exit(1)
    *src, dst = sys.argv[1:]
    merge_route_xml(src, dst)