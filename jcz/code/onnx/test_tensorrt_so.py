import tensorrt as trt, ctypes, os, glob

# 手动加载自定义插件 so
ctypes.CDLL("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/plugins/TensorRT/lib/libtensorrt_ops.so")

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")       # 初始化所有插件

registry = trt.get_plugin_registry()
plugin_names = [creator.name for creator in registry.plugin_creator_list]
print("registered plugins:", plugin_names)

# 直接查某个插件是否存在
rotate_creator = registry.get_plugin_creator("RotateTRT", "1", "")
print("RotateTRT registered?", rotate_creator is not None)