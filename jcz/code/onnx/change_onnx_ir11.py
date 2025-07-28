import onnx
from onnx import version_converter

OPT_VERSION = 15
IR_VERSION = 10

source_path = "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/ir10/uniad_tiny_imgx0.25_cp_b2d.onnx"
dist_path =   "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/ir10/uniad_tiny_imgx0.25_cp.ir10.onnx"

model = onnx.load(source_path)
model.ir_version = IR_VERSION
model = version_converter.convert_version(model, OPT_VERSION)
onnx.save(model, dist_path)