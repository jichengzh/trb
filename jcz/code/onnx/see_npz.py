import numpy as np
import numpy as np, sys, pathlib
from onnx import numpy_helper, mapping, TensorProto, shape_inference

npz_path = "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/calib_data_shape0_901.npz"      # 改成你的文件
npz = np.load(npz_path)                        # 默认 allow_pickle=False

# ① 查看里面有哪些数组（键名）
print("keys in npz:", list(npz.keys()), len(npz.files))

for key in npz.files:
    arr = npz[key]
    print(f"{key}: dtype={arr.dtype}, shape={arr.shape}")

# ④ 最后记得关闭（或用 with … as …）
npz.close()

import onnx
model = onnx.load("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.repaired.onnx")
print("input need in onnx", [t.name for t in model.graph.input])

for node in model.graph.node:
    if "RotateTRT" in node.op_type:
        print(f"Node: {node.name}, Inputs: {node.input}, Outputs: {node.output}")

# 创建输入输出类型的查找表
type_dict = {}
for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
    type_proto = vi.type.tensor_type
    shape = [d.dim_value if (d.dim_value > 0) else -1 for d in type_proto.shape.dim]
    dtype = mapping.TENSOR_TYPE_TO_NP_TYPE.get(type_proto.elem_type, "Unknown")
    type_dict[vi.name] = {"shape": shape, "dtype": dtype}

# 遍历查找 RotateTRT 节点
for node in model.graph.node:
    if "RotateTRT" in node.op_type:
        print(f"\nNode: {node.name} | Type: {node.op_type}")
        print("Inputs:")
        for inp in node.input:
            t = type_dict.get(inp, None)
            if t:
                print(f"  - {inp}: shape={t['shape']}, dtype={t['dtype']}")
            else:
                print(f"  - {inp}: (unknown or not inferred)")
        print("Outputs:")
        for out in node.output:
            t = type_dict.get(out, None)
            if t:
                print(f"  - {out}: shape={t['shape']}, dtype={t['dtype']}")
            else:
                print(f"  - {out}: (unknown or not inferred)")