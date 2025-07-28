import onnx

model = onnx.load("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/ir10/0_uniad_base_imgx0.25_cp_b2d.repaired.onnx")
graph = model.graph

# 1️⃣ 先把名字映射成 node / initializer，便于查询
output2node = {}       # 张量名 -> 生产它的 node
name2init   = {}       # 权重/常量 initializer

for n in graph.node:
    for out in n.output:
        output2node[out] = n
for init in graph.initializer:
    name2init[init.name] = init          # 常量 tensor

# 2️⃣ 找到 grid_sampler 节点
gs_node = next(n for n in graph.node if n.op_type == "grid_sampler")

print(">>> 当前 grid_sampler 节点:")
print("   name :", gs_node.name)
print("   inputs :", gs_node.input)
print("   outputs:", gs_node.output)

# 3️⃣ 打印每个 input 的来源
print("\n>>> 上游节点 / 权重:")
for inp in gs_node.input:
    if inp in name2init:
        print(f"  {inp:40s} <- initializer  (权重/常量)")
    elif inp in output2node:
        up = output2node[inp]
        print(f"  {inp:40s} <- {up.op_type:15s}  ({up.name})")
    else:
        print(f"  {inp:40s} <- ???  (graph input?)")

# 4️⃣ 打印每个 output 的去向
print("\n>>> 下游节点:")
for out in gs_node.output:
    sinks = [n for n in graph.node if out in n.input]
    if sinks:
        for dn in sinks:
            print(f"  {out:40s} -> {dn.op_type:15s}  ({dn.name})")
    else:
        print(f"  {out:40s} -> graph output / unused")
