# test_engine.py
import tensorrt as trt
# from cuda import cuda, nvrtc
import pycuda.driver as cuda
import pycuda.autoinit  # 这会自动初始化 CUDA 上下文
import numpy as np
import ctypes

ENGINE_PATH = "/home/featurize/Bench2DriveZoo/ckpts/engine/uniad_tiny_dummy_q_b2d_ganzhi_2.engine"
ctypes.CDLL("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/plugins/TensorRT/lib/libtensorrt_ops.so",
             mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL('/home/featurize/DL4AGX/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_plugin.so', 
            mode=ctypes.RTLD_GLOBAL)

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}"

    def __repr__(self):
        return self.__str__()

def load_engine(path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to load engine")
    print(f"[+] Loaded engine: {path}")
    return engine

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    # 获取所有张量名称
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    
    # 遍历所有张量
    for tensor_name in tensor_names:
        # 获取张量模式（INPUT 或 OUTPUT）
        tensor_mode = engine.get_tensor_mode(tensor_name)
        
        # 获取张量形状
        shape = engine.get_tensor_shape(tensor_name)
        # 处理动态维度 (-1 表示动态)
        if any(dim < 0 for dim in shape) and tensor_mode == trt.TensorIOMode.INPUT:
            # 获取配置文件中的最优形状
            profile_index = 0  # 默认使用第一个配置文件
            shape = engine.get_tensor_profile_shape(tensor_name, profile_index)[1]  # 1表示最优形状
        elif any(dim < 0 for dim in shape) and tensor_mode == trt.TensorIOMode.OUTPUT:
            # print("[DEBUG]", tensor_name, "has dynamic shape:", shape)
            if tensor_name.startswith("prev_track") or "prev_track" in tensor_name:
                # prev_track 相关的，用 901 填充所有动态维度
                shape = tuple(1150 if dim < 0 else dim for dim in shape)
            else:
                # 其余输出（检测/追踪头），用 81 填充所有动态维度
                shape = tuple(1150 if dim < 0 else dim for dim in shape)
            # print("[DEBUG]  -> allocate with shape", shape)
        
        # 计算体积并分配内存
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        # 分配固定内存和GPU内存
        host_mem = cuda.pagelocked_empty(int(size), dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # 添加到绑定列表
        bindings.append(int(device_mem))
        
        # 区分输入输出
        if tensor_mode == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
        
        # print(f"Tensor: {tensor_name}, Mode: {tensor_mode}, Shape: {shape}, DType: {dtype}, ")

        # print(len(inputs), len(outputs), len(bindings))
    
    # print(f"  Host Mem: {host_mem.shape}, Device Mem: {device_mem}")
    # print(f"  Total Inputs: {len(inputs)}, Total Outputs: {len(outputs)}")

       
    
    return inputs, outputs, bindings, stream, tensor_names

def do_inference(context, bindings, inputs, outputs, stream, data, tensor_names, engine):
    # 将输入数据从主机内存复制到GPU内存
    # for inp in inputs:
    #     cuda.memcpy_htod_async(inp.device, inp.host, stream)

    for i, (name, data) in enumerate(data.items()):
        input_shape = inputs[i].host.shape # 461312
        data_shape = data.shape
        # print(f"[DEBUG] Input {name}: data.shape = {data_shape}, expected = {input_shape}, flatten sizes: {data.flatten().shape[0]} vs {inputs[i].host.shape[0]}")
        context.set_input_shape(name, data_shape) 
        np.copyto(inputs[i].host, data.flatten())
        cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        context.set_tensor_address(name, inputs[i].device)
        
        # input_idx += 1
    # 给out数据分配gpu地址-----------------------------------------------------------------
    i = 0
    for j, tensor_name in enumerate(tensor_names):
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            # print('[DEBUG] Output tensor:', tensor_name)
            context.set_tensor_address(tensor_name, outputs[i].device)
            # print(f"[DEBUG] Set output tensor {tensor_name} to device address {outputs[i].device}")
            i += 1
    
    stream.synchronize()
    # 执行推理
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    # 将输出数据从GPU内存复制回主机内存
    results = []
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
        results.append(out.host)
    
    # 同步流以确保所有操作完成
    stream.synchronize()
    
    # 返回推理结果
    return results

def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
    # print(f"  · implicit batch: {engine.has_implicit_batch_dimension}")
    inputs, outputs, bindings, stream, tensor_names = allocate_buffers(engine)


    data = {
        'prev_track_intances0': np.zeros((901, 512), dtype=np.float32),
        'prev_track_intances1': np.zeros((901, 3), dtype=np.float32),
        'prev_track_intances3': np.zeros((901,), dtype=np.int32),
        'prev_track_intances4': np.zeros((901,), dtype=np.int32),
        'prev_track_intances5': np.zeros((901,), dtype=np.int32),
        'prev_track_intances6': np.zeros((901,), dtype=np.float32),
        'prev_track_intances8': np.zeros((901,), dtype=np.float32),
        'prev_track_intances9': np.zeros((901, 10), dtype=np.float32),
        'prev_track_intances11': np.zeros((901, 4, 256), dtype=np.float32),
        'prev_track_intances12': np.zeros((901, 4), dtype=np.int32),
        'prev_track_intances13': np.zeros((901,), dtype=np.float32),
        'prev_timestamp':         np.zeros((1,), dtype=np.float32),
        'prev_l2g_r_mat':         np.zeros((1, 3, 3), dtype=np.float32),
        'prev_l2g_t':             np.zeros((1, 3), dtype=np.float32),
        'prev_bev':               np.zeros((2500, 1, 256), dtype=np.float32),
        'timestamp':              np.zeros((1,), dtype=np.float32),
        'l2g_r_mat':              np.zeros((1, 3, 3), dtype=np.float32),
        'l2g_t':                  np.zeros((1, 3), dtype=np.float32),
        'img':                    np.zeros((1, 6, 3, 256, 416), dtype=np.float32),
        'img_metas_can_bus':      np.zeros((18,), dtype=np.float32),
        'img_metas_lidar2img':    np.zeros((1, 6, 4, 4), dtype=np.float32),
        'command':                np.zeros((1,), dtype=np.float32),
        'use_prev_bev':           np.zeros((1,), dtype=np.int32),
        'max_obj_id':             np.zeros((1,), dtype=np.int32),
    }


    out = do_inference(context, bindings, inputs, outputs, stream, data, tensor_names, engine)
    print("[+] Inference succeeded. Output tensors:")
    print(len(out))


if __name__ == "__main__":
    main()