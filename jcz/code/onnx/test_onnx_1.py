import onnxruntime as rt

def inspect_model(file_path="/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.onnx"):
    # 创建 InferenceSession
    sess = rt.InferenceSession(file_path)
    
    # 输出模型输入信息
    print("Model Inputs:")
    for inp in sess.get_inputs():
        print(f"  Name: {inp.name}")
        print(f"    Shape: {inp.shape}")
        print(f"    Type: {inp.type}")
    
    # 输出模型输出信息
    print("\nModel Outputs:")
    for outp in sess.get_outputs():
        print(f"  Name: {outp.name}")
        print(f"    Shape: {outp.shape}")
        print(f"    Type: {outp.type}")

if __name__ == "__main__":
    inspect_model()