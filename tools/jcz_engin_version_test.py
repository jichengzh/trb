def get_trt_serialization_version(engine_path):
    with open(engine_path, "rb") as f:
        magic = f.read(64)            # b'TRTE'
        ver   = int.from_bytes(f.read(4), "little")
        head = f.read(4)
    print("raw head bytes:", head)
    print("ASCII:", head[:4])
    print("serial (little-endian int):", int.from_bytes(head[4:8], "little"))

get_trt_serialization_version("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/onnx/uniad.engine")
# print("magic:", magic, "serialization ver:", ver)