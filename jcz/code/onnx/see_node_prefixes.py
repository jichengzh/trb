import onnx, re
from collections import Counter

def list_weight_prefixes(onnx_path, split_pat=r'[./]'):
    """
    列出 ONNX 中所有“权重张量”的名称前缀，及出现次数。

    Parameters
    ----------
    onnx_path : str   # 模型路径
    split_pat : str   # 用于切分前缀的正则，默认在 '.' 或 '/' 处分段

    Returns
    -------
    Counter({prefix: count, ...})
    """
    model = onnx.load(onnx_path)
    split_re = re.compile(split_pat)

    prefixes = []

    # ① 所有 initializer 都是权重 / 常量
    for init in model.graph.initializer:
        name = init.name
        prefix = split_re.split(name)[0]   # 取第一段
        prefixes.append(prefix)

    # ② Constant 节点的 'value' 也是隐式权重
    for node in model.graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                prefix = split_re.split(out)[0]
                prefixes.append(prefix)

    return Counter(prefixes)

# ---------- 用法示例 ----------
if __name__ == "__main__":
    onnx_file = "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/ir10/0_uniad_base_imgx0.25_cp_b2d.repaired.onnx"
    stats = list_weight_prefixes(onnx_file)

    print(f"共发现 {sum(stats.values())} 个权重张量，前缀分布：")
    for pfx, cnt in stats.most_common():
        print(f"{pfx:<20s} {cnt}")