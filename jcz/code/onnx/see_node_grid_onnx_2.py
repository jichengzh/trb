import onnx, onnx_graphsurgeon as gs
from typing import List, Optional, Tuple, Union

def get_prefix_of_node(model: Union[str, onnx.ModelProto],
                       node_name: str,
                       prefixes : Tuple[str, ...]) -> Optional[str]:
    """
    判断 `node_name` 属于哪一个模块前缀（若无匹配返回 None）

    Parameters
    ----------
    model     : .onnx 路径或已加载 ModelProto
    node_name : 节点的完整 name，如 "/bev_sampler/grid_sampler"
    prefixes  : ('pts_bbox_head.','seg_head.','motion_head.','occ_head.','planning_head.')

    Returns
    -------
    匹配到的前缀字符串，或 None
    """
    if isinstance(model, str):
        model = onnx.load(model)
    g = gs.import_onnx(model)

    # ① 找到目标节点
    node = next((n for n in g.nodes if n.name == node_name), None)
    if node is None:
        raise ValueError(f"Node {node_name} not found")

    # ② 检查所有输入 / 输出张量名
    for t in (*node.inputs, *node.outputs):
        for p in prefixes:
            if t.name.startswith(p):
                return p       # 命中即返回
    return None

prefixes = ('pts_bbox_head.','seg_head.','motion_head.','occ_head.','planning_head.','img_backbone.','img_neck.')
onnx_path = "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/ir10/0_uniad_base_imgx0.25_cp_b2d.repaired.onnx"
gs_name   = "/bev_sampler/grid_sampler"

p = get_prefix_of_node(onnx_path, gs_name, prefixes)
print(f"{gs_name} 属于模块: {p}")