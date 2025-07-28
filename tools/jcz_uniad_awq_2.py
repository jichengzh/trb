import sys, os
import sys, pathlib
bad = pathlib.Path('/data1/jcz/DL4AGX/AV-Solutions/uniad-trt/UniAD/third_party').resolve()
sys.path[:] = [p for p in sys.path if not pathlib.Path(p).resolve().is_relative_to(bad)]
# sys.path.insert(0, "/data1/jcz/AutoAWQ")   # 让 awq1 所在目录加入搜索路径
import torch, awq, json, os
from awq.quantize.quantizer import AwqQuantizer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from projects.awq_quantize.jcz_quantize_bevformer import UniversalAwqQuantizer
from projects.awq_quantize.jcz_uniad_awq import UniADAWQModel

from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset
from mmcv import Config, DictAction
import argparse
import warnings
from mmcv.runner import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmcv.runner import load_checkpoint
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from awq.quantize.quantizer import pseudo_quantize_tensor
# from awq.modules.linear import (
#     WQLinear_GEMM,
#     WQLinear_GEMV,
#     WQLinear_Marlin,
#     WQLinear_GEMVFast,
# )
# import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # plugin机制，允许通过config文件定义的自定义python代码目录，从而在不修改源代码的前提下扩展功能
    # （如：mmdet3d_plugin）
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    # 设置每个gpu的样本数，这里设置为1，因为需要校准
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)


    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

        # 加载校准集
    dataset = build_dataset(cfg.data.test)
    calib_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )            # §2 提到的校准集
    # 校准模型

    # 加载模型
    # model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda().eval()                 # FP32 权重
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # target_submodule = model.pts_bbox_head                           # 子模块
    # 初始化量化器

    checkpoint=load_checkpoint(model, args.checkpoint, map_location='cpu')

        # 合并conv和bn层
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # 旧版的checkpoint没有保存类别信息，这个兼容性处理是为了向后兼容
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES
    # # 处理语义分割的配色
    # # palette for visualization in segmentation tasks
    # if 'PALETTE' in checkpoint.get('meta', {}):
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    # elif hasattr(dataset, 'PALETTE'):
    #     # segmentation dataset has `PALETTE` attribute
    #     model.PALETTE = dataset.PALETTE



#----------------------dataloader数据解包--------------------------------
    # from mmcv.parallel.scatter_gather import scatter_kwargs
    # from mmcv.parallel.scatter_gather import scatter
    # calib_samples = []
    # for i, batch_data in enumerate(calib_loader):
    #     calib_samples.append(batch_data)
    #     if i >= 10:  # 只取前10个batch作为校准数据
    #         break
    # device_id = 1
    # device = torch.device('cuda:1')
    # scattered_samples = []
    # for batch_data in calib_samples:
    #     scattered_batch = scatter(batch_data, [-1])
    #     scattered_samples.append(scattered_batch[0])  # 取设备1上的数据
    # print(f"Type of scattered_samples: {type(scattered_samples)}")
    # print(f"Type of scattered_samples[0]: {type(scattered_samples[0])}")
    # print(f"Keys in scattered_samples[0]: {list(scattered_samples[0].keys())}")
    # print(type(scattered_samples[0]['img'][0]))

    # imgs = []
    # for batch in scattered_samples:
    #     img = batch['img'][0]  # 提取 tensor
    #     imgs.append(img)
    # print('type(imgs): ', type(imgs))
    # print('len(imgs): ', len(imgs))
    model = MMDataParallel(model, device_ids=[0])





#---------------------------------量化--------------------------------
    # model.cuda()
    # model.eval()
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    awq_backend = UniADAWQModel(
        model=model,
        model_type="jcz_uniad_awq",
        is_quantized=False,
        config=cfg.model,
        quant_config=quant_config,
        processor=None,
    )
    quantizer = UniversalAwqQuantizer(
        awq_backend=awq_backend,     # 或你为 DETR/BEVFormer 写的 backend
        model=model,
        calib_data=calib_loader, # 校准集
        tokenizer=None,
        w_bit=4, group_size=128, version="gemm",
        quant_config=quant_config,
        duo_scaling=True,  # 禁用 duo-scaling
        apply_clip=True,   # 禁用 clipping
        quant_layers=["seg_head"]
    )

    # quantizer.quantize()
    # 修改量化模块到UniversalAwqQuantizer里面修改该self.modules
    # 可供压缩的层包括:["motion_head", "seg_head", "occ_head", "pts_bbox_head", "planning_head"]
    # quantizer = UniversalAwqQuantizer(
    #     awq_backend=awq_backend, 
    #     model=model,
    #     # target_submodule=target_submodule,
    #     tokenizer=None,
    #     version="gemm",  # 或 "marlin"
    #     w_bit=8, group_size=128, zero_point=True,
    #     # 禁用 duo-scaling / clipping（虽然 pack_only 不会用到，也写上）
    #     duo_scaling=False,
    #     apply_clip=False,
    #     quant_config=quant_config,
    #     quant_layers=["occ_head"],
    #     # [ "motion_head", "seg_head", "occ_head", "pts_bbox_head", "planning_head"],
    # )

    # 不跑 self.quantize()，而是：
    quantizer.quantize()

    quantizer.save_quantized("/home/featurize/output/Uniad-quant-jcz", fmt="pth")

    # torch.save(model.state_dict(), "model_awq_int4.pth")
  



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    # import debugpy
    # debugpy.listen(12361)
    # print('wait debugger')
    # debugpy.wait_for_client()
    # print("Debugger Attached")
    main()

    