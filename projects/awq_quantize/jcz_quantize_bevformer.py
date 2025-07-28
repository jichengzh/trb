# universal_awq_quantizer.py
import functools, logging, inspect, math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import transformers   # 只为兼容 HuggingFace prepare_inputs_for_generation

# --- ↓↓↓ 直接沿用 AWQ 官方工具  ↓↓↓ -----------------------------------------
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import clear_memory, get_best_device
from awq.modules.linear import (
    WQLinear_GEMM, WQLinear_GEMV, WQLinear_Marlin, WQLinear_GEMVFast
)
from awq.utils.module import (
    append_str_prefix, get_op_name, get_named_linears,
    set_op_by_name, exclude_layers_to_not_quantize
)
from typing_extensions import Doc, Annotated
from huggingface_hub import snapshot_download, save_torch_state_dict
import os
from mmcv.runner import save_checkpoint
# ---------------------------------------------------------------------------

__all__ = ["UniversalAwqQuantizer"]

class UniversalAwqQuantizer:
    r"""
    通用 AWQ 量化器：
      · 保留官方 AWQ 搜索 / 缩放 / 裁剪 / 打包逻辑
      · 不假定模型层次结构——只要内部用了 nn.Linear，就能量化
    """
    QLINEAR_MAP = {
        "gemm":      WQLinear_GEMM,
        "gemv":      WQLinear_GEMV,
        "marlin":    WQLinear_Marlin,
        "gemv_fast": WQLinear_GEMVFast,
    }

    # ---------- 初始化 ------------------------------------------------------
    def __init__(
        self,
        awq_backend,               # awq.modeling.xxx，实现 get_model_layers / move_embed ...
        model: nn.Module,
        # target_submodule: nn.Module,  # 目标子模块，通常是 transformer block
        tokenizer,
        w_bit: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: str = "gemm",
        # ----- 数据&流程 -----
        calib_data=None,
        split: str = "train",
        text_column: str = "text",
        max_calib_samples: int = 128,
        max_calib_seq_len: int = 512,
        n_parallel_calib_samples: Optional[int] = None,
        # ----- 选项 -----
        duo_scaling: bool = True,
        apply_clip: bool = True,
        export_compatible: bool = False,
        modules_to_not_convert: Optional[List[str]] = None,
        max_chunk_mem_bytes: int = 1 << 30,  # 二进制数1左移三十位相当于2^30=1024*1024*1024
        quant_config=None,  # 量化配置，包含了量化的参数
        device = torch.device('cuda:1'),
        quant_layers: Optional[List[str]] = None,  # 量化的层名列表，若为None则量化所有层
    ):
        self.device            = device
        self.model             = model.eval()
        # self.target_submodule = target_submodule
        self.tokenizer         = tokenizer
        self.backend           = awq_backend
        self.w_bit             = w_bit
        self.group_size        = group_size
        self.zero_point        = zero_point
        self.version           = version
        self.calib_data        = calib_data
        self.split             = split
        self.text_column       = text_column
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.duo_scaling       = duo_scaling
        self.apply_clip        = apply_clip
        self.export_compatible = export_compatible
        self.max_chunk_memory= max_chunk_mem_bytes
        self.quant_layers = quant_layers or []
        # self.max_chunk_memory  = max_chunk_memory
        self.modules_to_not_convert = modules_to_not_convert or []
        self.modules = self.backend.get_model_layers(self.model, self.quant_layers)
        self.quant_config = quant_config

        # ↓ 初始化：拉出所有 transformer block、以及校准样本 / 输入缓存
        # 其中的layer_kwargs是第一层模块的输入参数，inps是第一层模块的输入数据. 所以这些数据可以保证精准不用进行再次校准
        # self.modules, self.layer_kwargs, self.inps = self._init_quant_data()

    # ---------- Public API --------------------------------------------------
    def quantize(self):
        """
        主入口：遍历所有 block → 搜索 scale / clip → 替换为 WQLinear_*。
        """
        print('------------------------------------------------------------------------------------')
        print(self.modules)
        for idx, layer in enumerate(self.modules, 1):
            print(f"正在处理第 {idx} 个 block：{get_op_name(self.model, layer)}")
            self._process_single_layer(idx, layer)
            clear_memory()

    def pack_only(self):
        """
        针对已完成 scale / clip 搜索、但权重尚未真正 pack 的模型；
        仅执行 `_apply_quant` 把 nn.Linear → WQLinear_*。
        """
        # print(self.modules)
        quantized_count = 0  # 新增计数器
        for layer in self.modules:
            print(f"正在处理第 {quantized_count + 1} 个 nn.Linear 层：{get_op_name(self.model, layer)}")
            named_linears = get_named_linears(layer)
            print("当前层的线性层：", named_linears)
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert)
            quantized_count += len(named_linears)  # 统计本层要量化的线性层数
            self._apply_quant(layer, named_linears)
            clear_memory()
        print(f"总共量化了 {quantized_count} 个 nn.Linear 层。")

    # ---------- 内部：量化单个 block ----------------------------------------
    def _process_single_layer(self, idx: int, layer: nn.Module):
        # -- 识别需要量化的层 ------------------------------------------
        named_linears = get_named_linears(layer)
        print(f"第 {idx} 个 block 中的 nn.Linear 层：{named_linears}")
        # 排除不需要量化的线性层, 函数本身就是一个过滤功能。要过滤什么层是要手动输入的
        named_linears = exclude_layers_to_not_quantize(
            named_linears, self.modules_to_not_convert)
        print(f"过滤结束")
        # -- 捕获输入特征 (activation) ----------------------------------
        input_feat = self._capture_inputs(layer, named_linears)
        # clear_memory()
        print(f"第 {idx} 个 block 的输入特征")

        # -- 搜索 + 应用 scale -----------------------------------------
        module_cfg = self.backend.get_layers_for_scaling(
            layer, input_feat, self.layer_kwargs)
        scales_list = [
            self._search_best_scale(layer, **cfg)
            for cfg in module_cfg
        ]
        apply_scale(layer, scales_list, input_feat_dict=input_feat)
        scales_list = append_str_prefix(
            scales_list, f"{get_op_name(self.model, layer)}."
        )
        print(f"第 {idx} 个 block 的缩放因子 scales_list: {scales_list}")

        # -- 搜索 + 应用 clip ------------------------------------------
        if self.apply_clip:
            clip_list = self._search_best_clip(layer, named_linears, input_feat)
            apply_clip(layer, clip_list)
            clip_list = append_str_prefix(
                clip_list, f"{get_op_name(self.model, layer)}."
            )
        print(f"第 {idx} 个 block 的裁剪因子 clip_list: {clip_list if self.apply_clip else '未应用裁剪'}")  

        # -- 真正替换 nn.Linear → WQLinear ----------------------------
        if not self.export_compatible:
            self._apply_quant(layer, named_linears)

    # ---------- _apply_quant：核心打包 -------------------------------------
    def _apply_quant(self, layer: nn.Module,
                     named_linears: Dict[str, nn.Linear]):
        qlinear_cls = self.QLINEAR_MAP[self.version]
        device_best = get_best_device()
        print('------------------------------------------------------------------------------------')
        for name, lin in named_linears.items():
            lin.to(device_best).half()
            w_q, scales, zeros = self._pseudo_quant(lin.weight.data)

            # group 内转置适配 GEMM layout
            if self.version == "gemm":
                scales = scales.t().contiguous()
                if zeros is not None:
                    zeros = zeros.t().contiguous()

            q_linear = qlinear_cls.from_linear(
                linear=lin,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales, zeros=zeros
            )

            lin.cpu()
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
            clear_memory()

    # -----------------------------------------------------------------------
    # ↓↓↓ 以下辅助函数几乎直接沿用原 AwqQuantizer，做了少量命名/内存适配 ↓↓↓
    # -----------------------------------------------------------------------
    def _pseudo_quant(self, w: torch.Tensor):
        org_shape = w.shape
        if self.group_size > 0:
            assert org_shape[-1] % self.group_size == 0, \
                "in_features 必须能被 group_size 整除"
            w = w.reshape(-1, self.group_size)

        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            qmax = 2**self.w_bit - 1
            scale = (max_val - min_val).clamp(min=1e-5) / qmax
            zero  = (-torch.round(min_val / scale)).clamp(0, qmax)
            w_q   = (torch.clamp(torch.round(w/scale)+zero,0,qmax)-zero)*scale
            zero  = zero.view(org_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            qmax    = 2**(self.w_bit-1) - 1
            scale   = max_val / qmax
            zero    = None
            w_q     = torch.clamp(torch.round(w/scale), -qmax, qmax)*scale

        scale = scale.view(org_shape[0], -1)
        return w_q.reshape(org_shape), scale, zero

    # ---- 搜索 scale / clip，与原版保持一致 ----
    # 由于代码过长，这里省略实现（与原 AwqQuantizer 同名方法完全一致）
    # from awq.quantize.search import (
    #     _search_best_scale as _search_best_scale,
    #     _search_best_clip  as _search_best_clip
    # )
    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        """
        AWQ 的核心思想是：权重和激活值的不同通道具有不同的重要性，需要为每个通道找到最优的缩放因子来最小化量化损失。
        """
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

# [STEP 1]: Compute per-channel mean of normalised weights
        # 将多个相关层的权重合并处理
        # 计算每个权重通道的"相对重要性"
        # 得到权重通道的统计特征
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0) # 将所有层的权重拼接在一起
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size) # 将权重矩阵重塑为 [num_groups, group_size] 的形状
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6) # 组内归一化
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape) # 将权重矩阵重塑为原始形状
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0) # 计算每个输出通道的平均缩放因子
        print(f"权重特征均值 w_mean: {w_mean}")
        clear_memory(weight)

# [STEP 2]: Compute per-channel mean of the input activation with chunking  # 计算每个输入通道的平均激活幅度
        # 计算每个输入通道的平均激活幅度
        # 使用分块处理来节省内存
        # 得到激活值通道的统计特征
        # move inp to cpu to avoid memory leak 
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1]) # 移动到cpu，取绝对值，改变形状。 最后inp_flat的形状是[num_elements, num_channels]其中存储了每个通道的激活幅度
        num_elements = inp_flat.size(0) 
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))# 计算每个通道的平均激活幅度
        chunk_size = min(chunk_size, num_elements)# 确保分块大小不超过输入张量的元素数量

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        # 分块计算避免内存溢出
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device) # x_sum是每个通道的激活幅度总和

        x_mean = (x_sum / num_elements).to(inp.dtype)
        print(f"输入特征均值 x_mean: {x_mean}")
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        print("计算最佳缩放因子...")
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view) # 缩放权重
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view # 模拟量化得到误差round(w / scale) + zero_point
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()
    # 若你 fork 中没有独立文件，可直接复制原函数体进来。

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        print("计算损失...")
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)
    
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    # ---- 捕获输入特征 ---------------------------------------------------
    def _capture_inputs(self, layer: nn.Module,
                        named_linears: Dict[str, nn.Linear]):
        feats = defaultdict(list)

        def _hook(m, x, y, name):
            feats[name].append(x[0].detach().cpu())
        handles = [mod.register_forward_hook(
            functools.partial(_hook, name=n)) for n, mod in named_linears.items()]

        # 输入转到正确设备
        self.inps = self.inps.to(next(layer.parameters()).device)
        # 执行前向传播
        self.inps = self._forward_chunk(self.inps, layer, self.layer_kwargs)

        # 清楚注册的hook
        for h in handles: h.remove()

        def _cat(k, v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] > 0, f"{k} 没有激活数据"
            return x
        return {k: _cat(k, v) for k, v in feats.items()}

    # ---- Forward with (optionally) chunked inputs -------------------------
    @torch.no_grad()
    def _forward_chunk(self, x, layer, kwargs):
        if self.n_parallel_calib_samples is None:
            out = layer(x, **kwargs)
            return out if not isinstance(out, tuple) else out[0]
        outs = []
        for xb in torch.split(x, self.n_parallel_calib_samples):
            yb = layer(xb, **kwargs)
            outs.append(yb[0] if isinstance(yb, tuple) else yb)
        return torch.cat(outs, dim=0)
    
    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # 内存高效地通过所有校准样本，但只通过 n_parallel_calib_samples 个样本
            # but only n_parallel_calib_samples at a time
            module_output = []
            # 将输入张量 x 分割成多个小批量，每个小批量包含 n_parallel_calib_samples 个样本
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    # ---- 初始化：提取所有 block & 校准样本 -------------------------------
    def _init_quant_data(self) -> Tuple[List[nn.Module], Dict, torch.Tensor]:
        modules = self.backend.get_model_layers(self.model)
        # -- 准备校准数据 --------------------------------------------------
        # samples = get_calib_dataset(
        #     data=self.calib_data, tokenizer=self.tokenizer,
        #     n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len,
        #     split=self.split, text_column=self.text_column
        # )
        # samples = torch.cat(samples, dim=0)      # B x seq_len
        # imgs = []
        # data_list = []
        # # <-- 传进来的 dataloader
        # for i in range(len(self.calib_data)):      # B x seq_len
        #     for key in self.calib_data[i].keys():


        #         data[key] = self.calib_data[i][key][0]
        

        #     data.append(data)
        # for sample in self.calib_data:          # 等价于 for i in range(len(self.calib_data))
        #     new_sample = {}
            
        #     for k, v in sample.items():
        #         # 如果 v 是 Tensor / list / tuple 且长度 ≥ 1，就取 v[0]；否则保持原样
        #         if isinstance(v, (list, tuple)):
        #             new_sample[k] = v[0]
        #         else:
        #             new_sample[k] = v          # 可能本来就是 Tensor / 标量 / 字典等
            
        #     data_list.append(new_sample)

        # device_best = get_best_device()

        # for i in range(len(self.calib_data)):      # B x seq_len

        #     img = self.calib_data[i]["img"][0]
        
        #     # 打印 img 的基本信息
        
        #     print(f"img type: {type(img)}")
        #     print(f"img length (if list): {len(img) if isinstance(img, list) else 'Not a list'}")

        #     # print(f"data keys: {list(self.calib_data.keys())}")
        #     # print(img)          # [B, N_cam, 3, H, W] 已按 pipeline 处理
        #     imgs.append(img)

        
        # calib_imgs = torch.cat(imgs, dim=0).to(device_best)  # [N, N_cam, 3, H, W]
        # -- 捕获第一层输入 & kwargs --------------------------------------
        # print(data_list)
        # print(type(data_list))
        # print(type(data_list[0]))
        # print(data_list[0].keys())
        # print(data_list[0]["img_metas"])

        # print(data_list[0]["img_metas"][0].keys())
        # print(data_list[0]["img_metas"][0])
        # print(data_list[0]["img_metas"][0][0])

        # inps, layer_kwargs = self._catch_first_layer_inputs(modules, data_list[0])
        inps, layer_kwargs = self._catch_first_layer_inputs(modules, self.calib_data)
        return modules, layer_kwargs, inps

    def _catch_first_layer_inputs(self, modules, samples): # samples = self.calib_data = dataloader
        """
        这个函数使用了一个巧妙的"拦截"技术：
        临时替换第一个模块为 Catcher
        执行一次前向传播来触发 Catcher
        在 Catcher 中捕获输入数据和参数
        恢复原始模块并清理资源
        """
        inps = []
        layer_kwargs = {}

        # -- 模型参数移动到最好的设备 --------------------------------------
        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.backend.move_embed(self.model, best_device)

        # -- 捕获第一层输入 & kwargs --------------------------------------
        class Catcher(nn.Module):
            def __init__(self, mod):
                super().__init__(); self.mod = mod
            def forward(self, *args, **kw):
                x = args[0] if args else kw.pop(list(kw.keys())[0])
                # 将输入张量 x 添加到 inps 列表中， 将kwargs添加到layer_kwargs字典中
                inps.append(x); layer_kwargs.update(kw); raise ValueError
        # 将第一层模块包装成一个Catcher模块，在Catcher模块的forward方法中，会调用self.mod的forward方法，并返回一个ValueError
        modules[0] = Catcher(modules[0])
        # try: self.model(return_loss=False, rescale=True, img=samples.to(next(self.model.parameters()).device))
        try:
            for i, data in enumerate(samples):
                with torch.no_grad():
                    self.model(return_loss=False, rescale=True, **data) 
            # self.model(return_loss=False, rescale=True, **samples)
        # 捕获到 ValueError 后，
        except ValueError: pass
        # 恢复第一层模块
        modules[0] = modules[0].mod                      # restore Layer0

        layer_kwargs = self.model.prepare_inputs_for_generation(
            samples, **layer_kwargs)
        layer_kwargs.pop("input_ids", None)
        inps = inps[0]

        modules[0].cpu(); self.backend.move_embed(self.model, "cpu")
        clear_memory()
        return inps, layer_kwargs
    
    def save_quantized(self, save_dir, fmt="pth", shard_size="5GB"):
        import os
        import torch
        import json
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        if fmt == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(self.model.state_dict(), os.path.join(save_dir, "model.safetensors"))
                print(f"Model saved in safetensors format to {save_dir}")
            except ImportError:
                print("safetensors not available, falling back to PyTorch format")
                torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        else:
            fname = "model.pth" if fmt == "pth" else "pytorch_model.bin"
            torch.save(self.model.state_dict(), os.path.join(save_dir, fname))
        
        meta = dict(                   # 你想额外存什么信息都可以塞进 meta
            quant_cfg = {
                'w_bit': self.w_bit,
                'group_size': self.group_size,
                'zero_point': self.zero_point,
                'version': self.version
            }
        )
        save_checkpoint(
            self.model,
            filename=os.path.join(save_dir, "model_quantized.pth"),
            optimizer=None,            # 如需一并保存优化器，可传 self.optimizer
            meta=meta,
            # create_symlink=False       # 不再额外建 latest.pth 软链
        )
        
        # 保存量化配置信息
        quant_config = {
            'w_bit': self.w_bit,
            'group_size': self.group_size,
            'zero_point': self.zero_point,
            'version': self.version
        }
        
        with open(os.path.join(save_dir, "quantization_config.json"), "w") as f:
            json.dump(quant_config, f, indent=2)
        
        print(f"Quantization config saved to {save_dir}/quantization_config.json")




    # def save_quantized(
    #     self,
    #     save_dir: Annotated[str, Doc("The directory to save your model to.")],
    #     safetensors: Annotated[
    #         bool, Doc("Whether to save the model as safetensors or torch files.")
    #     ] = True,
    #     shard_size: Annotated[
    #         str, Doc("The shard size for sharding large models into multiple chunks.")
    #     ] = "5GB",
    # ):
    #     save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

    #     # Save model
    #     class EmptyModule(nn.Module):
    #         def __init__(self):
    #             super(EmptyModule, self).__init__()

    #         def forward(self, x):
    #             return x

    #     # Save model and config files with empty state dict
    #     # 保存量化配置
    #     # self.model.config.quantization_config = self.quant_config
    #     # self.model.generation_config.do_sample = True
    #     self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

    #     # Vision transformers have a processor
    #     if self.processor is not None:
    #         self.processor.save_pretrained(save_dir)

    #     # Remove empty state dict
    #     default_paths = [
    #         f"{save_dir}/model.safetensors",
    #         f"{save_dir}/pytorch_model.bin",
    #     ]
    #     for path in default_paths:
    #         if os.path.exists(path):
    #             os.remove(path)

    #     save_torch_state_dict(
    #         state_dict=self.model.state_dict(),
    #         save_directory=save_dir,
    #         max_shard_size=shard_size,
    #         safe_serialization=safetensors,
    #         force_contiguous=True,
    #         shared_tensors_to_discard=self.model._tied_weights_keys,
    #     )
    



# from awq import AutoAWQForCausalLM
# from transformers import AutoTokenizer

# model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
# quant_path = 'mistral-instruct-v0.2-awq'
# quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# # Load model
# model = AutoAWQForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # Quantize
# model.quantize(tokenizer, quant_config=quant_config)

# # Save quantized model
# model.save_quantized(quant_path)
# tokenizer.save_pretrained(quant_path)

# print(f'Model is quantized and saved at "{quant_path}"')