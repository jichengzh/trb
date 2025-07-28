import os
import torch
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from mmcv import Config
from mmcv.runner import load_checkpoint
from third_party.uniad_mmdet3d.models.builder import build_model

# --------- 你需要手动设定的路径 ------------
CFG_PATH = '/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py'
CKPT_PATH = '/home/featurize/Bench2Drive/Bench2DriveZoo/ckpts/uniad_tiny_b2d.pth'
ONNX_OUT = './onnx/ir10/0_uniad_tiny_imgx0.25_cp_b2d.onnx'
INPUT_NPY_ROOT = './nuscenes_np/uniad_onnx_input/'      # 如果已有 .npy
CACHED_PREV_OUT_ROOT = './nuscenes_np/uniad_pth_trtp_out/'  # 如果要读取上一帧输出

# onnx_folder = "./onnx/ir10/"
# folder_dat = './dumped_inputs/'
# onnx_file_name = onnx_folder+"0_uniad_tiny_imgx0.25_cp_b2d.onnx"
# onnx_export_input = './nuscenes_np/uniad_onnx_input/'
# onnx_export_output = './nuscenes_np/uniad_pth_trtp_out/'
# -------------------------------------------

# 根据文件名推断分辨率（沿用你原来的逻辑，可自行硬编码）
def infer_shapes_from_name(name):
    if 'tiny' in name:
        bevh = 50
        img_h, img_w = 480, 800
        if 'x0.25' in name:
            img_h, img_w = 256, 416
    else:
        bevh = 100
        img_h, img_w = 928, 1600
    return bevh, img_h, img_w

bevh, img_h, img_w = infer_shapes_from_name(ONNX_OUT)
device = 'cuda'

# 与原脚本一致的输入名及其占位 shape（包含 -1 的动态维，仅用于辅助）
input_shapes = dict(
    prev_track_intances0 = [-1, 512],
    prev_track_intances1 = [-1, 3],
    prev_track_intances2 = [-1, 256],
    prev_track_intances3 = [-1],
    prev_track_intances4 = [-1],
    prev_track_intances5 = [-1],
    prev_track_intances6 = [-1],
    prev_track_intances7 = [-1],
    prev_track_intances8 = [-1],
    prev_track_intances9 = [-1, 10],
    prev_track_intances10 = [-1,10],
    prev_track_intances11 = [-1, 4, 256],
    prev_track_intances12 = [-1, 4],
    prev_track_intances13 = [-1],
    prev_timestamp = [1],
    prev_l2g_r_mat = [1,3,3],
    prev_l2g_t = [1,3],
    prev_bev = [bevh**2, 1, 256],
    gt_lane_labels = [1, -1],
    gt_lane_masks = [1, -1, bevh*2, bevh*2],
    gt_segmentation = [1, 7, bevh*2, bevh*2],
    img_metas_scene_token = [32],
    timestamp = [1],
    l2g_r_mat = [1,3,3],
    l2g_t = [1,3],
    img = [1,6,3,img_h,img_w],
    img_metas_can_bus = [18],
    img_metas_lidar2img = [1,6,4,4],
    image_shape = [2],
    command = [1],
    use_prev_bev = [1],
    max_obj_id = [1],
)

# 输出名字（顺序要与真实 forward 返回的 tuple 顺序一致）
output_names = [
    'prev_track_intances0_out',
    'prev_track_intances1_out',
    'prev_track_intances3_out',
    'prev_track_intances4_out',
    'prev_track_intances5_out',
    'prev_track_intances6_out',
    'prev_track_intances8_out',
    'prev_track_intances9_out',
    'prev_track_intances11_out',
    'prev_track_intances12_out',
    'prev_track_intances13_out',
    'prev_timestamp_out',
    'prev_l2g_t_out',
    'prev_l2g_r_mat_out',
    'bev_embed',
    'bboxes_dict_bboxes',
    'scores',
    'labels',
    'bbox_index',
    'obj_idxes',
    'max_obj_id_out',
    'seg_out',
    'outs_planning'
]

# 需要动态的轴（如果你想彻底固定，也可以删除）
dynamic_axes = {k:[0] for k in input_shapes if k.startswith('prev_track_intances')}
dynamic_axes.update({
    'prev_track_intances0_out':[0],
    'prev_track_intances1_out':[0],
    'prev_track_intances3_out':[0],
    'prev_track_intances4_out':[0],
    'prev_track_intances5_out':[0],
    'prev_track_intances6_out':[0],
    'prev_track_intances8_out':[0],
    'prev_track_intances9_out':[0],
    'prev_track_intances11_out':[0],
    'prev_track_intances12_out':[0],
    'prev_track_intances13_out':[0],
    'bboxes_dict_bboxes':[0],
    'scores':[0],
    'labels':[0],
    'bbox_index':[0],
    'obj_idxes':[0],
})

# --------- 构造 / 读取 输入张量 -----------
def load_or_dummy(name, shape):
    """
    如果你已经提前准备了 INPUT_NPY_ROOT/<name>/final.npy 就读取；
    否则根据约定生成 0 / 合理的默认值。
    """
    npy_dir = os.path.join(INPUT_NPY_ROOT, name)
    candidate = os.path.join(npy_dir, 'final.npy')
    if os.path.isfile(candidate):
        arr = np.load(candidate)
        tensor = torch.from_numpy(arr)
    else:
        # 简单的 dummy 策略：把 -1 维替换成 0（可按需要调成固定最大值）
        concrete = []
        for d in shape:
            if d == -1:
                # 给一个 0 行会导致某些操作失败，如果需要最小值可用 1
                concrete.append(0 if name.startswith('prev_track_intances') else 1)
            else:
                concrete.append(d)
        # 避免 0 维度导致导出不通过，可以统一替换成 1
        if any(dim == 0 for dim in concrete):
            concrete = [1 if dim == 0 else dim for dim in concrete]
        tensor = torch.zeros(concrete, dtype=torch.float32)

        # 一些特殊输入的 dtype / 值
        if name in ['command','use_prev_bev','max_obj_id']:
            tensor = tensor.to(dtype=torch.int32)
        if name.startswith('prev_track_intances') and len(tensor.shape)==1:
            tensor = tensor.to(dtype=torch.int32)
        if name in ['img_metas_scene_token']:
            tensor = torch.zeros(concrete, dtype=torch.int64)
        if name == 'image_shape':
            tensor = torch.tensor([img_h, img_w], dtype=torch.float32)

    return tensor.to(device)

# 构建保持“输入次序”的列表与名字（保证与原 torch.onnx.export 的 input_names 对应）
input_names = list(input_shapes.keys())
input_tensors = [load_or_dummy(k, input_shapes[k]) for k in input_names]

# --------- 构建模型并加载权重 -----------
cfg = Config.fromfile(CFG_PATH)
cfg.model.pretrained = None
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, CKPT_PATH, map_location='cpu')
model = model.cuda().eval()
# 指定 forward
model.forward = model.forward_uniad_trt

# --------- 导出 ONNX -----------
os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)

with torch.no_grad():
    torch.onnx.export(
        model,
        tuple(input_tensors),
        ONNX_OUT,
        input_names=input_names,
        output_names=output_names,
        opset_version=15,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        dynamic_axes=dynamic_axes,
    )

# --------- 修补 Reshape allowzero -----------
graph = gs.import_onnx(onnx.load(ONNX_OUT))
for node in graph.nodes:
    if node.op == "Reshape":
        node.attrs["allowzero"] = 1
onnx.save(gs.export_onnx(graph), ONNX_OUT[:-5] + "_repaired.onnx")

print("Export finished:", ONNX_OUT)