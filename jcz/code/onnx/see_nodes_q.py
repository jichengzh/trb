import re, onnx, onnx_graphsurgeon as gs


def find_node_by_name(model, prefixes):
    if isinstance(model, str):
        model = onnx.load(model)

    g = gs.import_onnx(model)

    # prefix = "planning_head."          # 你想量化的模块前缀
    #对每个节点 n，把它所有的 输入张量 (n.inputs) 和 输出张量 (n.outputs) 拼到一起。
    #只要这些张量里 有任何一个 的 name 以 prefix 开头 (startswith(prefix))，就说明节点与该模块相关。
    prefixes = tuple(prefixes)             # 方便 startswith 一次匹配
    target_nodes = {
        n.name
        for n in g.nodes
        if any(t.name.startswith(prefixes) for t in (*n.inputs, *n.outputs))
    }
    # print(target_nodes)
    print(f"{len(target_nodes)} nodes collected")
    return target_nodes

import onnx
import onnx_graphsurgeon as gs


def collect_seg_head_range(onnx_path, seg_prefixes=("seg_head.",)):
    model = onnx.load(onnx_path)
    g = gs.import_onnx(model)

    # 确保 tuple 格式
    if isinstance(seg_prefixes, str):
        seg_prefixes = (seg_prefixes,)

    nodes = g.nodes

    def uses_any_prefix(node):
        for t in (*node.inputs, *node.outputs):
            if t.name and t.name.startswith(seg_prefixes):
                return True
        return False

    seg_indices = [i for i, n in enumerate(nodes) if uses_any_prefix(n)]
    if not seg_indices:
        print(f"未找到任何前缀 {seg_prefixes} 命中节点，需检查是否正确")
        return [], (None, None)

    lo, hi = min(seg_indices), max(seg_indices)
    coarse_block = nodes[lo:hi+1]

    # 命中率检查
    hits = sum(uses_any_prefix(n) for n in coarse_block)
    ratio = hits / len(coarse_block)
    print(f"[区间范围] index = [{lo}, {hi}] 节点数 = {len(coarse_block)}，直接命中前缀的 {hits} 个，命中率 = {ratio:.3f}")

    return coarse_block, (lo, hi)

def combine_blocks(onnx_path, prefixes):
    model = onnx.load(onnx_path)
    g = gs.import_onnx(model)
    nodes = g.nodes

    def uses_any_prefix(node, pf_tuple):
        return any(
            t.name and t.name.startswith(pf_tuple)
            for t in (*node.inputs, *node.outputs)
        )

    def collect_coarse_block_for_prefix(pf):
        pf_tuple = (pf,) if isinstance(pf, str) else tuple(pf)
        seg_indices = [i for i, n in enumerate(nodes) if uses_any_prefix(n, pf_tuple)]
        if not seg_indices:
            print(f"⚠ 未找到任何前缀 {pf_tuple}")
            return []
        lo, hi = min(seg_indices), max(seg_indices)
        coarse_block = nodes[lo:hi + 1]
        hits = sum(uses_any_prefix(n, pf_tuple) for n in coarse_block)
        ratio = hits / len(coarse_block)
        print(f"[{pf_tuple}] 区间=[{lo},{hi}] 节点数={len(coarse_block)}, 命中={hits}, 命中率={ratio:.3f}")
        return coarse_block

    all_blocks = []
    if isinstance(prefixes, (list, tuple)):
        for pf in prefixes:
            all_blocks.extend(collect_coarse_block_for_prefix(pf))
    else:
        all_blocks = collect_coarse_block_for_prefix(prefixes)

    # ✅ 用节点 name 去重，保持顺序
    seen = set()
    combined_unique = []
    for n in all_blocks:
        if n.name not in seen:
            combined_unique.append(n)
            seen.add(n.name)

    print(f"✅ 合并后节点总数: {len(combined_unique)} (去重后)")
    return combined_unique

def filter_nodes_by_node_name(onnx_path, prefixes):
    """
    仅根据节点自身 name 的前缀进行筛选
    """
    import onnx
    import onnx_graphsurgeon as gs

    model = onnx.load(onnx_path)
    g = gs.import_onnx(model)
    nodes = g.nodes

    def match_node_name(node, pf_tuple):
        return node.name and node.name.startswith(pf_tuple)

    pf_tuple = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    matched_nodes = [n for n in nodes if match_node_name(n, pf_tuple)]

    print(f"🎯 仅按节点名匹配前缀 {pf_tuple} → 共命中 {len(matched_nodes)} 个节点")
    return matched_nodes

def filter_nodes_by_tensor_name(onnx_path, prefixes):
    """
    仅根据节点的输入/输出张量名的前缀进行筛选
    """
    import onnx
    import onnx_graphsurgeon as gs

    model = onnx.load(onnx_path)
    g = gs.import_onnx(model)
    nodes = g.nodes

    def match_tensor_name(node, pf_tuple):
        return any(
            t.name and t.name.startswith(pf_tuple)
            for t in (*node.inputs, *node.outputs)
        )

    pf_tuple = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    matched_nodes = [n for n in nodes if match_tensor_name(n, pf_tuple)]

    print(f"🎯 按张量名匹配前缀 {pf_tuple} → 共命中 {len(matched_nodes)} 个节点")
    return matched_nodes


def combine_blocks2(onnx_path, prefixes, export_hits=False):
    model = onnx.load(onnx_path)
    g = gs.import_onnx(model)
    nodes = g.nodes

    def uses_any_prefix(node, pf_tuple):
        return any(
            t.name and t.name.startswith(pf_tuple)
            for t in (*node.inputs, *node.outputs)
        )

    def collect_coarse_block_for_prefix(pf):
        pf_tuple = (pf,) if isinstance(pf, str) else tuple(pf)
        seg_indices = [i for i, n in enumerate(nodes) if uses_any_prefix(n, pf_tuple)]
        # print(seg_indices)
        if not seg_indices:
            print(f"⚠ 未找到任何前缀 {pf_tuple}")
            return [], []

        lo, hi = min(seg_indices), max(seg_indices)
        coarse_block = nodes[lo:hi + 1]

        # ✅ 区间里真正命中的节点
        hit_nodes = [n for n in coarse_block if uses_any_prefix(n, pf_tuple)]
        hit_nodes_names = [n.name for n in coarse_block if uses_any_prefix(n, pf_tuple)]

        hits = len(hit_nodes)
        ratio = hits / len(coarse_block)
        print(f"[{pf_tuple}] 区间=[{lo},{hi}] 节点数={len(coarse_block)}, 命中={hits}, 命中率={ratio:.3f}")

        return coarse_block, hit_nodes, hit_nodes_names

    # === 最终统计 ===
    all_blocks = []
    all_hits = []

    if isinstance(prefixes, (list, tuple)):
        for pf in prefixes:
            block, hits, hits_names = collect_coarse_block_for_prefix(pf)
            all_blocks.extend(block)
            all_hits.extend(hits)
            if export_hits:  # 可选保存每个block的命中节点
                with open(f"hits_{pf}.txt", "w") as f:
                    f.write("\n".join([n.name for n in hits]))
                print(f"✅ 命中节点已保存 hits_{pf}.txt")
    else:
        block, hits, hits_names = collect_coarse_block_for_prefix(prefixes)
        all_blocks = block
        all_hits = hits
        if export_hits:
            with open(f"hits_{prefixes}.txt", "w") as f:
                f.write("\n".join([n.name for n in hits]))
            print(f"✅ 命中节点已保存 hits_{prefixes}.txt")

    # 去重合并整个 all_blocks
    seen = set()
    combined_unique = []
    for n in all_blocks:
        if n.name not in seen:
            combined_unique.append(n)
            seen.add(n.name)

    print(f"✅ 合并后区间节点总数: {len(combined_unique)} (去重后)")
    print(f"✅ 所有命中节点总数: {len(all_hits)}")

    # 返回两类结果：
    # 1. 合并的所有区间节点
    # 2. 直接命中的节点（hit）
    return combined_unique, all_hits, hits_names

onnx_path = '/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.repaired_simp.onnx'
# 用法
block= combine_blocks2(
    '/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.repaired_simp.onnx',
    prefixes = ('pts_bbox_head','seg_head','motion_head','occ_head','planning_head')
)
# nodes = filter_nodes_by_tensor_name(onnx_path, ("/img_backbone", "/img_neck"))
# nodes = filter_nodes_by_node_name(onnx_path, "/img_backbone")
# nodes = filter_nodes_by_tensor_name(onnx_path, ("/pts_bbox_head", "/seg_head",'/motion_head','/occ_head','/planning_head', '/img_backbone', '/img_neck'))
# prefixes1 = ('pts_bbox_head','seg_head','motion_head','occ_head','planning_head', 'img_backbone', 'img_neck')
# prefixes2 = ('img_neck',)
# prefixes3 = ('pts_bbox_head','seg_head','motion_head','occ_head','planning_head', )
# nodes_to_exclude2 = find_node_by_name(onnx_path, prefixes1)
# nodes_to_exclude2 = find_node_by_name(onnx_path, prefixes2)
# nodes_to_exclude2 = find_node_by_name(onnx_path, prefixes3)


def extract_block_names_by_range(onnx_path, perception_range=(55, 4558), planning_range=(4558, 7745)):
    # 1. 加载 ONNX 并获取节点列表
    model = onnx.load(onnx_path)
    g = gs.import_onnx(model)
    nodes = g.nodes

    # 2. 索引区间
    lo_p, hi_p = perception_range
    lo_plan, hi_plan = planning_range

    # 3. 提取节点名称
    perception_names = [n.name for n in nodes[lo_p:hi_p]]
    planning_names   = [n.name for n in nodes[lo_plan:hi_plan]]

    print(f"✅ Perception block: idx=[{lo_p},{hi_p}] 节点数={len(perception_names)}")
    print(f"✅ Planning block:   idx=[{lo_plan},{hi_plan}] 节点数={len(planning_names)}")

    return perception_names, planning_names

extract_block_names_by_range(onnx_path)