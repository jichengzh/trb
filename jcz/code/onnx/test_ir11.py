import onnx

model_path = "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp_b2d_ort_support.onnx"
model = onnx.load(model_path)

print("IR version:", model.ir_version)
print("Opset imports:")
for op in model.opset_import:
    print(f"  - domain: '{op.domain}', version: {op.version}")