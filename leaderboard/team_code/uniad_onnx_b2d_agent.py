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
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes, glob

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
        engine_path = "/data1/jcz/DL4AGX/AV-Solutions/uniad-trt/onnx/uniad_tiny_dummy.engine"

        # 加载插件
        ctypes.CDLL('/data1/jcz/DL4AGX/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_plugin.so')

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 创建 TensorRT 运行时(runtime) 对象，用于反序列化 engine
        runtime = trt.Runtime(TRT_LOGGER)

        
        # 反序列化engine字节流，得到可执行的ICudaEngine 实例
        self.engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        # 创建 执行上下文，用来在运行时绑定输入输出 shape、执行推理。一个 engine 可以创建多个 context 以支持多流并发。
        self.context = self.engine.create_execution_context()
        print(self.engine.has_implicit_batch_dimension)
        
        
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding_idx))
            # 若使用显式批处理，volume需要乘以批大小。此处假设engine已包含批维度在binding shape中
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            # 分配page-locked内存
            host_mem = cuda.pagelocked_empty(int(size), dtype)
            # 分配GPU内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # 保存指针用于execute_async_v2
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding_idx):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
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
                    'sensor_tick': 0.04,
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
    def _preprocess_image(self, image):
        """
        将CARLA相机的原始图像数据转换为模型输入的numpy数组，并进行归一化预处理。
        """
        # 将carla.Image的BGRA 8位数据转成numpy数组
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        img_array = img_array[:, :, :3]  # 取前三通道(BGR)
        # 转为float32
        img_array = img_array.astype(np.float32)
        # 减去均值（使用UniAD训练时的图像归一化均值）:contentReference[oaicite:4]{index=4}
        # UniAD模型使用BGR顺序的均值103.53, 116.28, 123.675，标准差为1:contentReference[oaicite:5]{index=5}
        img_array[:, :, 0] -= 103.53  # B通道减均值
        img_array[:, :, 1] -= 116.28  # G通道减均值
        img_array[:, :, 2] -= 123.675 # R通道减均值
        # 不需要除以std因为std为1，且不需要转换RGB顺序(to_rgb=False)
        # 转换shape为CHW格式
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array

    def tick(self, input_data):
        """
        处理传感器输入数据，返回模型所需的numpy数据。
        input_data: 字典，键为传感器ID，值为(tuple(timestamp, data))
        """
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
        images = np.stack(images, axis=0)  # shape: (6, 3, H, W)
        
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
                'imgs': images,
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
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
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

        results['img'] = tick_data['imgs']

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
        results = self.inference_only_pipeline(results)

        # ========== TensorRT推理部分 ==========
        # 准备TensorRT输入数据
        trt_inputs = self._prepare_tensorrt_inputs(results)
        
        # 执行TensorRT推理
        trt_outputs = self._run_tensorrt_inference(trt_inputs)
        
        # 解析TensorRT输出
        output_data_batch = self._parse_tensorrt_outputs(trt_outputs)

        # -- 车辆控制和规划 -------------------------------------------------
        out_truck = output_data_batch[0]['planning']['result_planning']['sdc_traj'][0]
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
        return control

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
        l2g_r_mat = results['l2g_r_mat'].flatten().astype(np.float32)
        l2g_t = results['l2g_t'].astype(np.float32)
        
        # 命令
        command = np.array([results['command']], dtype=np.float32)
        
        # CAN总线数据
        can_bus = results['can_bus'].astype(np.float32)
        
        # 相机内外参
        lidar2img = results['lidar2img'].flatten().astype(np.float32)
        
        # 是否使用上一帧BEV特征（首帧设为0）
        use_prev_bev = np.array([0.0 if self.step <= 0 else 1.0], dtype=np.float32)
        
        return {
            'img': img_data,
            'timestamp': timestamp,
            'l2g_r_mat': l2g_r_mat,
            'l2g_t': l2g_t,
            'command': command,
            'img_metas_can_bus': can_bus,
            'img_metas_lidar2img': lidar2img,
            'use_prev_bev': use_prev_bev
        }

    def _run_tensorrt_inference(self, inputs):
        """
        执行TensorRT推理
        """
        # 将输入数据复制到GPU
        input_idx = 0
        for key, data in inputs.items():
            if input_idx < len(self.inputs):
                np.copyto(self.inputs[input_idx].host, data.flatten())
                cuda.memcpy_htod_async(self.inputs[input_idx].device, self.inputs[input_idx].host, self.stream)
                input_idx += 1
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 获取输出
        outputs = {}
        for i, output in enumerate(self.outputs):
            cuda.memcpy_dtoh_async(output.host, output.device, self.stream)
        
        # 同步流
        self.stream.synchronize()
        
        # 将输出数据转换为需要的格式
        # 这里需要根据你的模型输出格式进行调整
        for i, output in enumerate(self.outputs):
            output_name = f"output_{i}"  # 根据实际输出名称调整
            outputs[output_name] = output.host.copy()
        
        return outputs

    def _parse_tensorrt_outputs(self, trt_outputs):
        """
        解析TensorRT输出，转换为与原PyTorch模型兼容的格式
        """
        # 这里需要根据你的具体输出格式进行解析
        # 假设主要输出是规划轨迹
        
        # 根据你的模型输出结构调整
        planning_output = trt_outputs.get('output_0', np.zeros((1, 6, 2)))  # 假设输出形状
        
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
        
        return output_data_batch

    def save(self, tick_data):
        frame = self.step // 10
        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
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
