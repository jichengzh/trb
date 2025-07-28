import onnx
import onnx, onnx_graphsurgeon as gs, re
import re, onnx, onnx_graphsurgeon as gs
from typing import List, Union
from collections import Counter



model = onnx.load("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/ir10/uniad_tiny_imgx0.25_cp_b2d.onnx")

model_path = "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/ir10/uniad_tiny_imgx0.25_cp_b2d.onnx"

def print_node_info(model):
    for n in model.graph.node:
        print(n.name, n.op_type)


def find_node_by_name(model, name):

    g = gs.import_onnx(model)

    prefix = "pts_bbox_head."          # 你想量化的模块前缀
    #对每个节点 n，把它所有的 输入张量 (n.inputs) 和 输出张量 (n.outputs) 拼到一起。
    #只要这些张量里 有任何一个 的 name 以 prefix 开头 (startswith(prefix))，就说明节点与该模块相关。
    target_nodes = [
        n.name for n in g.nodes
        if any(t.name.startswith(prefix) for t in n.inputs + n.outputs)
    ]
    print(f"{len(target_nodes)} nodes collected")





def find_node_by_name2(model: Union[str, onnx.ModelProto],
                      prefix: str) -> List[str]:
    """
    收集与 `prefix` 前缀张量相连、且 **不是偏置节点** 的节点名称。

    Args:
        model:  .onnx 路径或已加载 ModelProto
        prefix: 希望量化的模块前缀，如 "pts_bbox_head."

    Returns:
        target_nodes: 可量化节点名列表（已排除 bias 相关）
    """
    _BIAS_QDQ_PAT  = re.compile(r".*\.bias_(QuantizeLinear|DequantizeLinear)$")
    _BIAS_TENSOR_PAT = re.compile(r".*\.bias$")      # 张量名以 ".bias" 结尾
    # 允许直接传路径
    if isinstance(model_path, str):
        model = onnx.load(model_path)

    graph = gs.import_onnx(model)
    target_nodes = []

    for n in graph.nodes:
        # 判断是否属于 prefix
        related = any(t.name.startswith(prefix) for t in n.inputs + n.outputs)
        if not related:
            continue

        # ---------- ❶ 排除 bias 量化节点 ----------
        #   /xxx.bias_QuantizeLinear  或  /xxx.bias_DequantizeLinear
        if _BIAS_QDQ_PAT.match(n.name):
            continue

        # ---------- ❷ 排除仅做 "加偏置" 的 Add 节点 ----------
        # 规则：节点是 Add，并且 **其中一个输入张量** 名字以 ".bias" 结尾
        if n.op == "Add" and any(_BIAS_TENSOR_PAT.match(t.name) for t in n.inputs):
            continue

        # 其它节点保留
        target_nodes.append(n.name)

    print(f"[INFO] {len(target_nodes)} nodes collected (bias nodes skipped)")
    return target_nodes

def find_node_by_name3(model, prefixes):
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

# find_node_by_name(model, "pts_bbox_head.")

# find_node_by_name2(model, "pts_bbox_head")

# find_node_by_name3(model, ["seg_head.","occ_head.","motion_head.","planning_head."])
# find_node_by_name3(model, ["seg_head."])
# find_node_by_name3(model, ["occ_head."])
# find_node_by_name3(model, ["planning_head."])
# find_node_by_name3(model, ["motion_head."])


#  ----------------------- 查看onnx模型里面的节点分布 ---------------------------------
def analyze_onnx_nodes(model_path):
    model = onnx.load(model_path)
    nodes = model.graph.node

    # 统计节点类型
    node_types = [node.op_type for node in nodes]
    counter = Counter(node_types)

    print(f"🔍 模型总节点数: {len(nodes)}")
    print("📊 节点类型分布:")
    for op_type, count in counter.most_common():
        print(f"  {op_type:15s} {count}")

    # 计算量化比例
    quantizable_ops = {"Conv", "Gemm", "MatMul", "Add", "MaxPool"}
    quantizable_count = sum(counter[op] for op in quantizable_ops)
    print(f"\n✅ 理论可量化节点数: {quantizable_count} ({quantizable_count/len(nodes)*100:.2f}%)")

analyze_onnx_nodes(model_path)