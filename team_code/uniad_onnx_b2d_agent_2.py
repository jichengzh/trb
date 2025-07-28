import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from Bench2DriveZoo.team_code.pid_controller import PIDController
from Bench2DriveZoo.team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.optimize import fsolve
SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

import tensorrt as trt
# from cuda import cuda, nvrtc
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes, glob
print("TRT runtime ver:", trt.__version__)
def get_entry_point():
    return 'UniadOnnxAgent'

# 定义一个辅助类，用于保存主机和设备上的buffer指针
class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}"

    def __repr__(self):
        return self.__str__()

class UniadOnnxAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.prev_track = {} # 初始化 prev_track 字典，用于存储前一帧的跟踪信息

        self.track = autonomous_agent.Track.SENSORS #设置当前 agent 的 track 类型为 SENSORS，表明该 agent 依赖传感器输入
        self.steer_step = 0 # 初始化方向盘转角步数
        self.last_moving_status = 0 # 初始化移动状态
        self.last_moving_step = -1 # 初始化移动步数
        self.last_steers = deque() # 初始化方向盘转角队列
        self.pidcontroller = PIDController() # 初始化PID控制器

        # 这两个配置文件未必会用到，但是为了保证程序正常运行，先保留
        self.config_path = path_to_conf_file.split('+')[0] # 初始化配置文件路径
        self.ckpt_path = path_to_conf_file.split('+')[1]  # 初始化监测点路径


        if IS_BENCH2DRIVE: 
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1 # 初始化步数
        self.wall_start = time.time() # 初始化时间
        self.initialized = False # 初始化标志
        cfg = Config.fromfile(self.config_path)
        cfg.model['motion_head']['anchor_info_path'] = os.path.join('Bench2DriveZoo',cfg.model['motion_head']['anchor_info_path'])
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  


        # # 配置文件和模型加载
        # engine_path = "/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.engine"
        # engine_path = "/home/featurize/Bench2Drive/Bench2DriveZoo/ckpts/engine/uniad_tiny_dummy_q_b2d.engine"
        engine_path = "/home/featurize/Bench2DriveZoo/ckpts/engine/only_ganzhi_uniad_tiny_dummy_q_b2d.engine"

        # 加载插件
        ctypes.CDLL('/home/featurize/DL4AGX/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_kernel.so', mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL('/home/featurize/DL4AGX/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_plugin.so', mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/plugins/TensorRT/lib/libtensorrt_ops.so",
             mode=ctypes.RTLD_GLOBAL)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 创建 TensorRT 运行时(runtime) 对象，用于反序列化 engine
        runtime = trt.Runtime(TRT_LOGGER)

        # print(1)
        # 反序列化engine字节流，得到可执行的ICudaEngine 实例
        self.engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        # 创建 执行上下文，用来在运行时绑定输入输出 shape、执行推理。一个 engine 可以创建多个 context 以支持多流并发。
        self.context = self.engine.create_execution_context()
        # print(self.engine.has_implicit_batch_dimension)
        
        
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.output_names = []
        self.stream = cuda.Stream()
        # 获取所有张量名称
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        # 遍历所有张量
        for tensor_name in tensor_names:
            # 获取张量模式（INPUT 或 OUTPUT）
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            # 获取张量形状
            shape = self.engine.get_tensor_shape(tensor_name)
            # 处理动态维度 (-1 表示动态)
            if any(dim < 0 for dim in shape) and tensor_mode == trt.TensorIOMode.INPUT:
                # print(1, tensor_name, "has dynamic shape:", shape)
                # 获取配置文件中的最优形状
                profile_index = 0  # 默认使用第一个配置文件
                # 用于获取某个张量在某个 profile 下的动态 shape 配置。
                # shape = self.engine.get_tensor_profile_shape(tensor_name, profile_index)[2]  # 1表示最优形状
                shape = tuple(1150 if dim < 0 else dim for dim in shape)
            elif any(dim < 0 for dim in shape) and tensor_mode == trt.TensorIOMode.OUTPUT:
                # print("[DEBUG]", tensor_name, "has dynamic shape:", shape)
                if tensor_name.startswith("prev_track") or "prev_track" in tensor_name:
                    # prev_track 相关的，用 901 填充所有动态维度
                    shape = tuple(1150 if dim < 0 else dim for dim in shape)
                else:
                    # 其余输出（检测/追踪头），用 81 填充所有动态维度
                    shape = tuple(1150 if dim < 0 else dim for dim in shape)

            # 计算体积并分配内存
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # 分配固定内存和GPU内存
            host_mem = cuda.pagelocked_empty(int(size), dtype)  # 通过pagelocked页锁定，在主机端分配固定内存区域 dtype表示指定的元素类型。
            device_mem = cuda.mem_alloc(host_mem.nbytes)  # mem_alloc在GPU端分配内存区域，返回一个指向设备内存的指针。
            
            # 添加到绑定列表
            self.bindings.append(int(device_mem))
            
            # 区分输入输出
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.output_names.append(tensor_name)
            
            # print(f"Tensor: {tensor_name}, Mode: {tensor_mode}, Shape: {shape}, DType: {dtype}, ")
         # 这里是不需要的，因为原版的uniada agent并不会把车辆 actor 直接塞给智能体。他们完全依靠SPEED 伪传感器来获取速度，依靠 GNSS / IMU 来获取位置与朝向。
        # try:
        #     self.vehicle = self._parent_vehicle  # 假设AutonomousAgent基类可能将ego vehicle赋给此属性
        # except AttributeError:
        #     self.vehicle = None
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        # compose是一个顺序容器，将多个transforms组合在一起   
        #  Compose 里的算子大多输出 torch.Tensor；而 TensorRT 部分期望的是 NumPy/FP32。若后面仍调用 self.inference_only_pipeline(results)，结果会是 DataContainer + torch.Tensor，与你在 run_step() 里直接 np.copyto() 不兼容。    
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        
        self.last_steers = deque()
        self.lat_ref, self.lon_ref = 42.0, 2.0
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            # string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string = self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
   
        # write extrinsics directly
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                                  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                       [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
         'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  ,  0.  , -1.  , -0.24],
                                 [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                      [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
          'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

        # 预热
        # self.context.execute_async_v2()

    # 重置流
    def reset_context(self):
        engine_path = "/home/featurize/Bench2DriveZoo/ckpts/engine/expect_imgbackbone_uniad_tiny_dummy_q_b2d.engine"

        # 加载插件
        ctypes.CDLL('/home/featurize/DL4AGX/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_kernel.so', mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL('/home/featurize/DL4AGX/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_plugin.so', mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("/home/featurize/DL4AGX/AV-Solutions/uniad-trt/UniAD/plugins/TensorRT/lib/libtensorrt_ops.so",
             mode=ctypes.RTLD_GLOBAL)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 创建 TensorRT 运行时(runtime) 对象，用于反序列化 engine
        runtime = trt.Runtime(TRT_LOGGER)

        # print(1)
        # 反序列化engine字节流，得到可执行的ICudaEngine 实例
        self.engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        print("[INFO] Reset TensorRT context+stream for new scenario")

    def _init(self):
        
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0        
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                
            ]
        
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors
    # def _preprocess_image(self, image):
    #     """
    #     将CARLA相机的原始图像数据转换为模型输入的numpy数组，并进行归一化预处理。
    #     """
    #     # 将carla.Image的BGRA 8位数据转成numpy数组
    #     img_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    #     img_array = img_array[:, :, :3]  # 取前三通道(BGR)
    #     # 转为float32
    #     img_array = img_array.astype(np.float32)
    #     # 减去均值（使用UniAD训练时的图像归一化均值）:contentReference[oaicite:4]{index=4}
    #     # UniAD模型使用BGR顺序的均值103.53, 116.28, 123.675，标准差为1:contentReference[oaicite:5]{index=5}
    #     img_array[:, :, 0] -= 103.53  # B通道减均值
    #     img_array[:, :, 1] -= 116.28  # G通道减均值
    #     img_array[:, :, 2] -= 123.675 # R通道减均值
    #     # 不需要除以std因为std为1，且不需要转换RGB顺序(to_rgb=False)
    #     # 转换shape为CHW格式
    #     img_array = np.transpose(img_array, (2, 0, 1))
    #     return img_array
    def _preprocess_image(self, image,
                      dst_wh=(400, 225),          # resize 目标分辨率 (W,H) = 1600×900 ×0.25
                      pad_multiple=32,            # 高、宽都 pad 到 32 的整倍
                      mean_bgr=(103.53, 116.28, 123.675),
                      std_bgr=(1., 1., 1.)):
        """
        把 CARLA 传回的 BGRA 图像加工成  float32 CHW，
        且分辨率 / padding / 归一化策略与 C++ TensorRT 侧保持一致
        -----------------------------------------------------------------
        返回: np.ndarray,  shape=(C, H_pad, W_pad), dtype=float32
        """
        # ① BGRA → BGR  &  uint8 → float32
        # print(f"[Debug] Got image of type: {type(image)}")
        if image.shape[2] == 4:
            image = image[..., :3]

        img = image.astype(np.float32)

        # ② 颜色归一化（先 /255，再减均值 / 除以 std）
        img = img / 255.0
        mean = np.asarray(mean_bgr, dtype=np.float32) / 255.0
        std  = np.asarray(std_bgr,  dtype=np.float32)
        img = (img - mean) / std          # 若 std 全为 1.0 等价于仅减均值

        # ③ resize 到 dst_wh（OpenCV 默认 BGR）
        dst_w, dst_h = dst_wh
        img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        # ④ padding 到 pad_multiple 的整数倍
        pad_h = int(np.ceil(dst_h / pad_multiple) * pad_multiple)
        pad_w = int(np.ceil(dst_w / pad_multiple) * pad_multiple)
        if pad_h != dst_h or pad_w != dst_w:
            padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32) # pad_h=256 pad_w=416
            padded[:dst_h, :dst_w] = img
            img = padded

        # ⑤ HWC → CHW
        img = img.transpose(2, 0, 1)      # (C=3, H_pad=256, W_pad=416)

        return img

    def tick(self, input_data):
        """
        处理传感器输入数据，返回模型所需的numpy数据。
        input_data: 字典，键为传感器ID，值为(tuple(timestamp, data))
        """
        self.step+=1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        # 提取并预处理所有摄像头图像
        cam_ids = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        images = []
        # imgs = {}
        # 注意设置runstep中的输入
        for cam_id in cam_ids:
            if cam_id in input_data:
                # 获取CARLA摄像头传感器数据对象
                image = input_data[cam_id][1]
                images.append(self._preprocess_image(image))
                # imgs[cam_id]=self._preprocess_image(image)
        images = np.stack(images, axis=0)  # shape: (6, 3, H=256, W=416)

        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        # imgs和images的区别就是前者是一个字典包括了每个摄像头的名称，后者是一个numpy数组，包含了所有摄像头的图像数据
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
        pos = self.gps_to_location(gps)

        # 匹配tensorrt的输入
        near_node, near_command = self._route_planner.run_step(pos)
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        
        result = {
                'imgs': imgs,  # 字典形式的图像数据
                'images': images,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
    
                }
        
        return result

    # def tick(self, input_data):
    #     self.step += 1
    #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
    #     imgs = {}
    #     for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
    #         img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
    #         _, img = cv2.imencode('.jpg', img, encode_param)
    #         img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #         imgs[cam] = img
    #     bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
    #     gps = input_data['GPS'][1][:2]
    #     speed = input_data['SPEED'][1]['speed']
    #     compass = input_data['IMU'][1][-1]
    #     acceleration = input_data['IMU'][1][:3]
    #     angular_velocity = input_data['IMU'][1][3:6]
    #     pos = self.gps_to_location(gps)
    #     near_node, near_command = self._route_planner.run_step(pos)
    #     if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
    #         compass = 0.0
    #         acceleration = np.zeros(3)
    #         angular_velocity = np.zeros(3)

    #     result = {
    #             'imgs': imgs,
    #             'gps': gps,
    #             'pos':pos,
    #             'speed': speed,
    #             'compass': compass,
    #             'bev': bev,
    #             'acceleration':acceleration,
    #             'angular_velocity':angular_velocity,
    #             'command_near':near_command,
    #             'command_near_xy':near_node
    
    #             }
        
    #     return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        # =====修改开始记录开始时间=====
        start_runstep_time = time.perf_counter()
        # =====修改结束=====
        # -- 初始化 ---------------------------------------------------
        if not self.initialized:
            self._init()
        # -- 数据预处理 -----------------------------------------------
        tick_data = self.tick(input_data)

        # -- 初始化结果字典 -------------------------------------------
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')

        # -- 多相机数据处理 -------------------------------------------------
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            # results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)

        results['img'] = tick_data['images']

        # -- 车辆姿态计算 -------------------------------------------------
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        # -- can_bus车辆状态信息 -------------------------------------------------
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus

        # -- 导航命令处理 -------------------------------------------------
        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        results['command'] = command

        # -- 局部坐标转换 -------------------------------------------------
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy

        # -- 全局坐标转换 -------------------------------------------------
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape

        # -- 模型输出顺序调整 -------------------------------------------------
        # results = self.inference_only_pipeline(results)

        # ========== TensorRT推理部分 ==========
        
        start_inference_time = time.perf_counter()
        # time.sleep(0.2)
        # 准备TensorRT输入数据
        
        trt_inputs = self._prepare_tensorrt_inputs(results)
        
        # 执行TensorRT推理
        trt_outputs = self._run_tensorrt_inference(trt_inputs)
        
        # 解析TensorRT输出
        output_data_batch = self._parse_tensorrt_outputs(trt_outputs)

        end_inference_time = time.perf_counter()
        inference_time = end_inference_time - start_inference_time
        # print(f"Inference time: {inference_time:.4f} seconds", flush=True)

            # ▼―――――――― 这里开始是新增 / 修改部分 ――――――――▼
        # 1. 取所有以 'prev_' 开头的键 + 'bev_embed'
        self.prev_track = {k: v for k, v in trt_outputs.items()
                    if k.startswith('prev_')}
        if 'bev_embed' in trt_outputs:          # 安全检查：有才加
            self.prev_track['bev_embed'] = trt_outputs['bev_embed']
        # ▲―――――――― 新增 / 修改结束 ――――――――▲


        # -- 车辆控制和规划 -------------------------------------------------
        out_truck = output_data_batch[0]['planning']['result_planning']['sdc_traj'][0]
        
        # print(out_truck)
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0
        if tick_data['speed']>5:
            throttle_traj = 0
        
        # -- 创建carla车辆控制对象 -------------------------------------------------
        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'tensorrt_uniad'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)

        # -- 元数据记录和保存 -------------------------------------------------
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if SAVE_PATH is not None and self.step % 1 == 0:
            self.save(tick_data)
        self.prev_control = control

        # =====修改开始， 记录运行时间=====
        end_runstep_time = time.perf_counter()
        runstep_time = end_runstep_time - start_runstep_time
        print(f"Run step time: {runstep_time:.4f} seconds", flush=True)
        return control, inference_time

    def _prepare_tensorrt_inputs(self, results):
        """
        准备TensorRT推理的输入数据
        数据形状调整：TensorRT引擎期望特定的输入形状，需要确保数据维度正确
        数据类型统一：确保所有输入都是float32类型
        内存布局：将多维数组展平为一维数组，符合TensorRT的内存布局要求
        时序信息：处理use_prev_bev等时序相关的输入
        """
        # 根据你的TensorRT模型输入要求，准备输入数据
        # 这里需要根据你的具体模型输入格式进行调整
        
        # 图像数据
        img_data = results['img'].astype(np.float32)  # shape: (6, 3, H, W)
        if len(img_data.shape) == 4:
            img_data = np.expand_dims(img_data, 0)  # 添加batch维度: (1, 6, 3, H, W)
        
        # 时间戳
        timestamp = np.array([results['timestamp']], dtype=np.float32)
        
        # 变换矩阵
        l2g_r_mat = results['l2g_r_mat'].astype(np.float32).reshape(1, 3, 3)
        l2g_t = results['l2g_t'].astype(np.float32).reshape(1, 3)
        
        # 命令
        command = np.array([results['command']], dtype=np.float32)
        
        # CAN总线数据
        can_bus = results['can_bus'].astype(np.float32)
        
        # 相机内外参
        lidar2img = results['lidar2img'].astype(np.float32).reshape(1, 6, 4, 4)  # 假设有6个相机
        
        # 是否使用上一帧BEV特征（首帧设为0）
        use_prev_bev = np.array([0 if self.step <= 0 else 1], dtype=np.int32)
        # use_prev_bev = np.array([0], dtype=np.int32)
        max_obj_id = np.array([1], dtype=np.int32)

        # 按照prev_track_intance0的输入配置----------------------------------------
    
    # --------------------------------------------------------
    # ▶▶ 工具函数：优先拿 self.prev_track，拿不到就全零
        def get_prev(key, shape, dtype):
            """key: 例如 'prev_track_intances0_out' or 'bev_embed'"""
            if self.prev_track and key in self.prev_track:
                return self.prev_track[key].astype(dtype)
            return np.zeros(shape, dtype=dtype)

        # ---- 逐个取 / 填充 ----
        prev_track_intances0  = get_prev('prev_track_intances0_out',  (1150, 512),       np.float32)
        prev_track_intances1  = get_prev('prev_track_intances1_out',  (1150, 3),         np.float32)
        prev_track_intances3  = get_prev('prev_track_intances3_out',  (1150,),           np.int32)
        prev_track_intances4  = get_prev('prev_track_intances4_out',  (1150,),           np.int32)
        prev_track_intances5  = get_prev('prev_track_intances5_out',  (1150,),           np.int32)
        prev_track_intances6  = get_prev('prev_track_intances6_out',  (1150,),           np.float32)
        prev_track_intances8  = get_prev('prev_track_intances8_out',  (1150,),           np.float32)
        prev_track_intances9  = get_prev('prev_track_intances9_out',  (1150, 10),        np.float32)
        prev_track_intances11 = get_prev('prev_track_intances11_out', (1150, 4, 256),    np.float32)
        prev_track_intances12 = get_prev('prev_track_intances12_out', (1150, 4),         np.int32)
        prev_track_intances13 = get_prev('prev_track_intances13_out', (1150,),           np.float32)

        prev_timestamp  = get_prev('prev_timestamp_out',  (1,),        np.float32)
        prev_l2g_r_mat  = get_prev('prev_l2g_r_mat_out',  (1, 3, 3),   np.float32)
        prev_l2g_t      = get_prev('prev_l2g_t_out',      (1, 3),      np.float32)
        prev_bev        = get_prev('bev_embed',           (2500, 1, 256), np.float32)
        # ---------------------------------------------------------------------
                
        return {
            'prev_track_intances0': prev_track_intances0,
            'prev_track_intances1': prev_track_intances1,
            'prev_track_intances3': prev_track_intances3,
            'prev_track_intances4': prev_track_intances4,
            'prev_track_intances5': prev_track_intances5,
            'prev_track_intances6': prev_track_intances6,
            'prev_track_intances8': prev_track_intances8,
            'prev_track_intances9': prev_track_intances9,
            'prev_track_intances11': prev_track_intances11,
            'prev_track_intances12': prev_track_intances12,
            'prev_track_intances13': prev_track_intances13,
            'prev_timestamp': prev_timestamp,
            'prev_l2g_r_mat': prev_l2g_r_mat,
            'prev_l2g_t': prev_l2g_t,
            'prev_bev': prev_bev,
            'timestamp': timestamp,
            'l2g_r_mat': l2g_r_mat,
            'l2g_t': l2g_t,
            'img': img_data,
            'img_metas_can_bus': can_bus,
            'img_metas_lidar2img': lidar2img,
            'command': command,
            'use_prev_bev': use_prev_bev,
            'max_obj_id':max_obj_id,


        }

    def _run_tensorrt_inference(self, inputs):
        """
        执行TensorRT推理
        """
        # 将输入数据复制到GPU
        input_idx = 0
        # 将输入数据拷贝到 GPU，并绑定张量地址
        for i, (name, data) in enumerate(inputs.items()):
            # 修改检查数据维度
            # print(f"[{i}] binding='{name}' host_dtype={self.inputs[i].host.dtype} " f"src_dtype={inputs[name].dtype}  shape={self.inputs[i].host.shape}")
            if self.inputs[i].host.dtype != inputs[name].dtype:
                data = data.astype(self.inputs[i].host.dtype)

            
            expected_shape = self.inputs[i].host.shape # 461312
            data_shape = data.shape
            # print(f"[DEBUG] Input {name}: data_shape={data_shape}, expected_host_shape={expected_shape}, "
            #     f"flatten sizes: data={data.size} vs expected={self.inputs[i].host.size}")
            if data.size != self.inputs[i].host.size:
                raise ValueError(f"[ERROR] Input {name} size mismatch! Got {data.size}, expected {self.inputs[i].host.size}")
            # print(f"[DEBUG] Input {name}: data.shape = {data_shape}, expected = {input_shape}, flatten sizes: {data.flatten().shape[0]} vs {self.inputs[i].host.shape[0]}")
            self.context.set_input_shape(name, data_shape) 
            np.copyto(self.inputs[i].host, data.flatten())
            cuda.memcpy_htod_async(self.inputs[i].device, self.inputs[i].host, self.stream)
            self.context.set_tensor_address(name, self.inputs[i].device)
            
            input_idx += 1
            
            # 绑定输出张量地址
        output_shapes = {}
        print(">>> Before execute_async_v3, current TRT binding shapes:")
        j = 0 
        for i, tensor_name in enumerate(self.output_names):
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                self.context.set_tensor_address(tensor_name, self.outputs[j].device)
                # print(f"[DEBUG] Set output tensor {tensor_name} to device address {self.outputs[j].device}")
                j += 1

        # 执行推理
        self.context.execute_async_v3(self.stream.handle)

        # print(f"[DEBUG] 执行推理，输入数量: {input_idx}, 输出数量: {len(self.outputs)}")
        self.stream.synchronize()
        # for output in self.outputs:
        #     cuda.memcpy_dtoh_async(output.host, output.device, self.stream)
        # 获取输出
        # 4. 拷贝输出并逐个校验
        for i, (name, out_mem) in enumerate(zip(self.output_names, self.outputs)):
            # print(f"[DEBUG] 输出 tensor `{name}` 的形状: {out_mem.host.shape}, 元素总数: {out_mem.host.size}")
            try:
                # 为定位问题，使用同步 memcpy
                cuda.memcpy_dtoh_async(out_mem.host, out_mem.device, self.stream)
            except cuda.LogicError as e:
                raise RuntimeError(
                    f"Memcpy failed for output '{name}' (index {i}), "
                    f"expected elements {out_mem.host.size}, "
                )


        # 同步流
        self.stream.synchronize()

        
        # 将输出数据转换为需要的格式
        # 这里需要根据你的模型输出格式进行调整
        outputs = {}
        # for i, output in enumerate(self.outputs):
        #     outputs[self.output_names[i]] = output.host.copy()
        for name, mem in zip(self.output_names, self.outputs):
            # shape = self.output_shapes[name]
            shape = self.context.get_tensor_shape(name)
            # 从一维 flat 数组恢复成对应的 multi‐dim
            outputs[name] = mem.host.reshape(shape).copy()
        
        # print(outputs)
        return outputs

    def _parse_tensorrt_outputs(self, trt_outputs):
        """
        解析TensorRT输出，转换为与原PyTorch模型兼容的格式
        """
        # 这里需要根据你的具体输出格式进行解析
        # 假设主要输出是规划轨迹
        
        # 根据你的模型输出结构调整
        planning_output = trt_outputs.get('outs_planning', np.zeros((1, 6, 2)))  # 假设输出形状
        
        # 重塑为期望格式
        if len(planning_output.shape) == 1:
            planning_output = planning_output.reshape(1, -1, 2)  # (batch, points, xy)
        
        # 构造与原PyTorch输出兼容的数据结构
        output_data_batch = [{
            'planning': {
                'result_planning': {
                    'sdc_traj': [planning_output[0]]  # 取第一个batch
                }
            }
        }]
        # print(output_data_batch)
        return output_data_batch

    def save(self, tick_data):
        if self.step % 10 != 0:
            return  # 只在每 10 帧一次保存
        frame = self.step // 10
        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
