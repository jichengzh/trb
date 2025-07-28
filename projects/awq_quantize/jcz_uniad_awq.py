import torch.nn as nn
from awq.models.base import BaseAWQForCausalLM

class UniADAWQModel(BaseAWQForCausalLM):
    """
    UniAD 模型的 AWQ 后端，实现对 UniAD 结构中各 Transformer 模块的量化支持。
    类似于 LlamaAWQForCausalLM，通过实现 get_model_layers, move_embed, get_layers_for_scaling 三个关键方法。
    """
    # 指定层类型（这里取UniAD中Transformer层名称作为标识，用于AWQ内部可能的融合逻辑）
    layer_type = "BEVFormerLayer"  # UniAD的主干Transformer层类型
    max_new_tokens_key = None  # 此模型无生成length限制相关参数

    @staticmethod
    def get_model_layers(model,
                     not_skip=["motion_head", "seg_head", "occ_head", "pts_bbox_head", "planning_head"],
                     group_size: int = 128) -> list[nn.Module]:
        """
        返回待量化的子模块列表。

        参数
        ----
        not_skip : set[str]       # 想整体跳过量化的子模块名，支持：
                            #   "motion_head", "seg_head", "occ_head",
                            #   "pts_bbox_head", "planning_head"
        group_size : int      # 用于 in_features 整除性检查
        """
        import torch.nn as nn
        from awq.utils.module import get_named_linears

        not_skip = not_skip or set()
        layers: list[nn.Module] = []

        def ok(mod: nn.Module) -> bool:
            """判断该子模块能不能量化：尺寸满足，且内部有 Linear。"""
            named_lins = get_named_linears(mod)
            if not named_lins:
                return False
            return all(w.weight.shape[1] % group_size == 0 for w in named_lins.values())

        # ----------- BEVFormer (pts_bbox_head) ---------------------------------
        if "pts_bbox_head" in not_skip and hasattr(model, "pts_bbox_head"):
            head = model.pts_bbox_head
            if hasattr(head, "transformer"):
                enc = head.transformer.encoder
                if hasattr(enc, "layers"):
                    layers.extend([blk for blk in enc.layers if ok(blk)])
                dec = head.transformer.decoder
                if hasattr(dec, "layers"):
                    layers.extend([blk for blk in dec.layers if ok(blk)])

        # ----------- SegHead ---------------------------------------------------
        if "seg_head" in not_skip and hasattr(model, "seg_head"):
            seg = model.seg_head.transformer
            if hasattr(seg, "encoder") and hasattr(seg.encoder, "layers"):
                layers.extend([blk for blk in seg.encoder.layers if ok(blk)])
            if hasattr(seg, "decoder") and hasattr(seg.decoder, "layers"):
                layers.extend([blk for blk in seg.decoder.layers if ok(blk)])

        # ----------- OccHead ---------------------------------------------------
        if "occ_head" in not_skip and hasattr(model, "occ_head"):
            occ = model.occ_head
            if hasattr(occ, "transformer_decoder") and hasattr(occ.transformer_decoder, "layers"):
                layers.extend([blk for blk in occ.transformer_decoder.layers if ok(blk)])

        # ----------- MotionHead -----------------------------------------------
        if "motion_head" in not_skip and hasattr(model, "motion_head"):
            mh = model.motion_head.motionformer
            if hasattr(mh, "intention_interaction_layers"):
                blk = mh.intention_interaction_layers.interaction_transformer
                if ok(blk): layers.append(blk)
            for lst_name in ["track_agent_interaction_layers",
                            "map_interaction_layers",
                            "bev_interaction_layers"]:
                for blk in getattr(mh, lst_name, []):
                    tgt = getattr(blk, "interaction_transformer", blk)
                    if ok(tgt): layers.append(tgt)

        # ----------- PlanningHead ---------------------------------------------
        if "planning_head" in not_skip and hasattr(model, "planning_head"):
            ph = model.planning_head
            if hasattr(ph, "attn_module") and hasattr(ph.attn_module, "layers"):
                layers.extend([blk for blk in ph.attn_module.layers if ok(blk)])

        return layers
    # def get_model_layers(model, skip: set[str] | None = None):
    #     """
    #     获取模型中所有需要量化的 Transformer 层模块列表。
    #     对于 UniAD，包括 BEVFormerEncoder 的各层、Detection Transformer Decoder 各层、
    #     分割 SegDeformableTransformer 的 Encoder/Decoder 各层，以及可能的 OccHead 和 MotionHead 的 Transformer层。
    #     返回按推理先后顺序排列的层模块列表。
    #     """
    #     layers = []
    #     # 1. BEVFormer Encoder 层 (BEVFormerLayer), 例如 model.pts_bbox_head.transformer.encoder.layers
    #     if hasattr(model, "pts_bbox_head"):
    #         head = model.pts_bbox_head

    #         # 1) Perception Transformer ------------------------------------------------
    #         if hasattr(head, "transformer"):
    #             # ─── Encoder 的 BEVFormerLayer 列表
    #             enc = head.transformer.encoder
    #             if hasattr(enc, "layers"):
    #                 layers.extend(list(enc.layers))          # list → extend

    #             # ─── Decoder：Detection / Perception Transformer Decoder
    #             dec = head.transformer.decoder
    #             if hasattr(dec, "layers"):
    #                 layers.extend(list(dec.layers))

    #             # can_bus_mlp 也是 Sequential(Linear···) ── 如需量化，可追加
    #             # if hasattr(head.transformer, "can_bus_mlp"):
    #             #     layers.append(head.transformer.can_bus_mlp)

    #         # 2) 检测分支（分类 / 2D-3D bbox / 过去轨迹） ------------------------------
    #         #    这些都是 ModuleList[Sequential(... Linear ...)] → 直接丢进 layers
    #         # for attr in [
    #         #     "cls_branches",
    #         #     "reg_branches",
    #         #     "past_traj_reg_branches",
    #         # ]:
    #         #     if hasattr(head, attr):
    #         #         layers.extend(list(getattr(head, attr)))   # ModuleList ⇒ list
    #     # 3. 地图分割 Pansegformer 的 SegDeformableTransformer Encoder/Decoder 层
    #     if hasattr(model, 'seg_head') and hasattr(model.seg_head, 'transformer'):
    #         seg_trans = model.seg_head.transformer
    #         # Encoder 层
    #         if hasattr(seg_trans, 'encoder') and hasattr(seg_trans.encoder, 'layers'):
    #             layers += list(seg_trans.encoder.layers)
    #         # Decoder 层
    #         if hasattr(seg_trans, 'decoder') and hasattr(seg_trans.decoder, 'layers'):
    #             layers += list(seg_trans.decoder.layers)
    #     # 4. 占用预测 OccHead 的 Transformer Decoder 层 (若存在)
    #     if hasattr(model, "occ_head"):
    #         occ = model.occ_head

    #         # 1) transformer_decoder 的每一层 (DetrTransformerDecoderLayer)
    #         if hasattr(occ, "transformer_decoder") and hasattr(occ.transformer_decoder, "layers"):
    #             layers += list(occ.transformer_decoder.layers)

    #         # 2) mode_fuser  / multi_query_fuser  / query_to_occ_feat  / temporal_mlp_for_mask
    #         #    这些都是 nn.Sequential 或 MLP，内部全是 Linear
    #         # for attr in [
    #         #     "mode_fuser",
    #         #     "multi_query_fuser",
    #         #     "query_to_occ_feat",
    #         #     "temporal_mlp_for_mask",
    #         # ]:
    #         #     if hasattr(occ, attr):
    #         #         layers.append(getattr(occ, attr))

            
    #     # 5. 轨迹预测 MotionHead 的 Transformer Decoder 层 (若存在)
    #     if hasattr(model, "motion_head"):

    #         m = model.motion_head.motionformer  # MotionTransformerDecoder

    #         # 1. IntentionInteraction —— EncoderLayer
    #         if hasattr(m, "intention_interaction_layers"):
    #             enc = m.intention_interaction_layers.interaction_transformer
    #             layers.append(enc)

    #         # 2. Track-Agent Interaction —— DecoderLayer * N
    #         if hasattr(m, "track_agent_interaction_layers"):
    #             for blk in m.track_agent_interaction_layers:
    #                 if hasattr(blk, "interaction_transformer"):
    #                     layers.append(blk.interaction_transformer)

    #         # 3. Map Interaction —— DecoderLayer * N
    #         if hasattr(m, "map_interaction_layers"):
    #             for blk in m.map_interaction_layers:
    #                 if hasattr(blk, "interaction_transformer"):
    #                     layers.append(blk.interaction_transformer)

    #         # 4. BEV Interaction —— MotionTransformerAttentionLayer * N
    #         #    直接把整块加入，里面 already 含 sampling_offsets/value_proj 等 Linear
    #         if hasattr(m, "bev_interaction_layers"):
    #             for blk in m.bev_interaction_layers:
    #                 layers.append(blk)

    #                 # ---------- ⑦ planning_head（PlanningHeadSingleMode） --------------------
    #     if hasattr(model, "planning_head"):
    #         ph = model.planning_head

    #         # 1) reg_branch  —— Sequential(Linear-ReLU-Linear)
    #         # if hasattr(ph, "reg_branch"):
    #         #     layers.append(ph.reg_branch)                  # Linear ×2

    #         # 2) attn_module：TransformerDecoder → 多层 TransformerDecoderLayer
    #         if hasattr(ph, "attn_module") and hasattr(ph.attn_module, "layers"):
    #             layers.extend(list(ph.attn_module.layers))

    #         # 3) mlp_fuser —— Sequential(Linear-LayerNorm-ReLU)
    #         # if hasattr(ph, "mlp_fuser"):
    #         #     layers.append(ph.mlp_fuser)
        

    #     return layers

    @staticmethod
    def move_embed(model, device: str):
        """
        将模型中的嵌入层参数移动到指定设备上。
        对 UniAD，需要移动多处嵌入或可学习参数到指定设备：
        - BEV 查询初始嵌入 (BEV query embedding)
        - 各 Transformer 解码器的查询嵌入 (例如 DETR目标查询embedding、分割查询embedding、OccHead查询embedding等)
        - 位置编码等可学习权重（如 BEV位置编码）
        这样在采样推理时，上述参数会位于正确的设备上。
        """
        # 1. BEV 初始查询嵌入 (UniAD 将BEV网格初始化为可学习嵌入)
        if hasattr(model, 'pts_bbox_head') and hasattr(model.pts_bbox_head, 'transformer'):
            trans = model.pts_bbox_head.transformer
            if hasattr(trans, 'bev_embedding'):  # BEV query embedding (nn.Embedding)
                model.pts_bbox_head.transformer.bev_embedding = trans.bev_embedding.to(device)
        # 2. Detection Head 目标查询嵌入
        if hasattr(model, 'pts_bbox_head'):
            if hasattr(model.pts_bbox_head, 'query_embedding'):  # DETR query embedding
                model.pts_bbox_head.query_embedding = model.pts_bbox_head.query_embedding.to(device)
            # 位置编码 (BEV positional encoding)
            if hasattr(model.pts_bbox_head, 'positional_encoding'):
                model.pts_bbox_head.positional_encoding = model.pts_bbox_head.positional_encoding.to(device)
        # 3. 分割 PansegformerHead 查询嵌入
        if hasattr(model, 'pansegformer_head'):
            if hasattr(model.pansegformer_head, 'query_embedding'):
                model.pansegformer_head.query_embedding = model.pansegformer_head.query_embedding.to(device)
        # 4. OccHead 查询嵌入
        if hasattr(model, 'occ_head'):
            if hasattr(model.occ_head, 'query_embedding'):
                model.occ_head.query_embedding = model.occ_head.query_embedding.to(device)
        # 5. MotionHead 查询嵌入或锚点表示
        if hasattr(model, 'motion_head'):
            if hasattr(model.motion_head, 'query_embedding'):
                model.motion_head.query_embedding = model.motion_head.query_embedding.to(device)
        return model

    @staticmethod
    def get_layers_for_scaling(layer, input_feat, module_kwargs):
        """
        给定一个 transformer 层模块 (encoder或decoder层)，返回该层中需要计算量化 scale 的线性层分组配置。
        每个返回的字典定义一组需要共享输入特征用于 scale 搜索的线性层:
        - 'prev_op': 该组线性层之前的操作(通常为layer norm)模块，用于获取正确的输入上下文
        - 'layers': 线性层列表 (nn.Linear) 需要一起量化 scale 计算
        - 'inp': 对应 input_feat 字典中这些层输入特征的键
        - 'module2inspect': （可选）若需要特殊处理的子模块，例如 multi-head attention 子模块，用于内部融合 QKV
        - 'kwargs': （可选）forward需要的额外参数，在处理多头注意力时传入 memory 等
        具体策略:
        * Self-Attention: 将 Q/K/V 投影层组合一起计算scale（若模型中QKV是分开的投影矩阵），这样可在输入embedding一致情况下统一确定量化比例。Attention的输出投影层单独为一组。
        * 若 Self-Attention 使用融合的 QKV 参数 (如 nn.MultiheadAttention 内部权重)，无法逐个获取q_proj/k_proj/v_proj子模块，则仅对其输出投影 out_proj 量化（此时 in_proj 留作FP16, 因无法直接拆分量化）。
        * Cross-Attention: 若存在，则类似处理。对于Deformable Attention这类多尺度注意力模块:
            - 将 offset 和 attention weight 两个线性层组合一起量化（输入同为query特征）。
            - Value投影 (value_proj) 与输出投影 (output_proj) 分开处理：value_proj 输入来自memory特征，output_proj输入来自attention输出。为简化流程，我们分别量化它们，并假定前一线性层的输出作为其输入上下文。
        * Feed-Forward网络 (FFN): 将前向第一层（expand层）和第二层（投影回embedding维度层）分别作为独立分组。第一层使用其输入（通常是上一子层输出经过Norm）的分布计算scale，第二层则使用第一层输出(激活后)的分布计算scale。
        返回: 包含一个层内所有线性层量化分组配置的列表。
        """
        layers = []
        # 1. Self-Attention input linear layers grouping
        if 'self_attn.q_proj' in input_feat:
            # 标准多头自注意力 (如LLaMA)：将 q_proj, k_proj, v_proj 作为一组
            prev_op = layers.input_layernorm if hasattr(layers, "input_layernorm") else nn.Identity()
            layers.append(
                dict(
                    prev_op=prev_op,
                    layers=[
                        layers.self_attn.q_proj,
                        layers.self_attn.k_proj,
                        layers.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=layers.self_attn,
                    kwargs=module_kwargs,
                )
            )
        elif 'self_attn.sampling_offsets' in input_feat:
            # Deformable 自注意力：将 sampling_offsets 和 attention_weights 分组
            prev_op = layers.norm1 if hasattr(layers, "norm1") else (layers.norm if hasattr(layers, "norm") else nn.Identity())
            layers.append(
                dict(
                    prev_op=prev_op,
                    layers=[
                        layers.self_attn.sampling_offsets,
                        layers.self_attn.attention_weights,
                    ],
                    inp=input_feat["self_attn.sampling_offsets"],
                    module2inspect=layers.self_attn,
                    kwargs=module_kwargs,
                )
            )
        elif 'self_attn.attn.out_proj' in input_feat:
            # PyTorch MultiheadAttention 情况：没有独立的 q_proj/k_proj/v_proj，仅 out_proj
            prev_op = layers.input_layernorm if hasattr(layers, "input_layernorm") else nn.Identity()
            layers.append(
                dict(
                    prev_op=prev_op,
                    layers=[layers.self_attn.attn.out_proj],
                    inp=input_feat["self_attn.attn.out_proj"],
                    # 对于单个线性层，这里不需要 module2inspect（AWQ会直接使用该层）
                )
            )
        # 2. Self-Attention output projection grouping
        if 'self_attn.o_proj' in input_feat:
            # 对于显式 o_proj：通常与 v_proj 形状相同，将其视为后一线性层
            if layers.self_attn.v_proj.weight.shape == layers.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=layers.self_attn.v_proj,
                        layers=[layers.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )
        elif 'self_attn.output_proj' in input_feat:
            # Deformable 自注意力的输出投影：与 value_proj 配对
            if hasattr(layers.self_attn, "value_proj") and layers.self_attn.value_proj.weight.shape == layers.self_attn.output_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=layers.self_attn.value_proj,
                        layers=[layers.self_attn.output_proj],
                        inp=input_feat["self_attn.output_proj"],
                    )
                )
        # 3. Cross-Attention input linear layers grouping (if decoder or multi-modal)
        if 'cross_attn.q_proj' in input_feat:
            # 标准交叉注意力：分组 q_proj, k_proj, v_proj
            # 前置层为经过自注意力后的LayerNorm（若存在），没有则用Identity占位
            prev_op = None
            if hasattr(layers, "post_attention_layernorm"):
                prev_op = layers.post_attention_layernorm
            elif hasattr(layers, "norm2"):
                prev_op = layers.norm2
            elif hasattr(layers, "norm"):
                prev_op = layers.norm
            else:
                prev_op = nn.Identity()
            layers.append(
                dict(
                    prev_op=prev_op,
                    layers=[
                        layers.cross_attn.q_proj,
                        layers.cross_attn.k_proj,
                        layers.cross_attn.v_proj,
                    ],
                    inp=input_feat["cross_attn.q_proj"],
                    module2inspect=layers.cross_attn,
                    kwargs=module_kwargs,
                )
            )
        if 'cross_attn.o_proj' in input_feat:
            if layers.cross_attn.v_proj.weight.shape == layers.cross_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=layers.cross_attn.v_proj,
                        layers=[layers.cross_attn.o_proj],
                        inp=input_feat["cross_attn.o_proj"],
                    )
                )
        if 'cross_attn.sampling_offsets' in input_feat:
            # Deformable 跨注意力：分组 sampling_offsets 和 attention_weights
            prev_op = layers.norms[1] if hasattr(layers, "norms") else nn.Identity()  # norm1 通常是跨注意力前的Norm
            layers.append(
                dict(
                    prev_op=prev_op,
                    layers=[
                        layers.cross_attn.sampling_offsets,
                        layers.cross_attn.attention_weights,
                    ],
                    inp=input_feat["cross_attn.sampling_offsets"],
                    module2inspect=layers.cross_attn,
                    kwargs=module_kwargs,
                )
            )
        if 'cross_attn.value_proj' in input_feat and 'cross_attn.output_proj' in input_feat:
            # Deformable跨注意力的值投影与输出投影
            if layers.cross_attn.value_proj.weight.shape == layers.cross_attn.output_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=layers.cross_attn.value_proj,
                        layers=[layers.cross_attn.output_proj],
                        inp=input_feat["cross_attn.output_proj"],
                    )
                )
        # 4. Feed-Forward network (FFN) layers grouping
        if 'mlp.gate_proj' in input_feat:
            # LLaMA类似的门控FFN：将 gate_proj 和 up_proj 分组
            layers.append(
                dict(
                    prev_op=layers.post_attention_layernorm,
                    layers=[layers.mlp.gate_proj, layers.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=layers.mlp,
                )
            )
        if 'mlp.down_proj' in input_feat:
            layers.append(
                dict(
                    prev_op=layers.mlp.up_proj,
                    layers=[layers.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )
        if 'ffn.layers.0.0' in input_feat:
            # UniAD标准FFN第一层：ffn.layers.0 内的线性层
            prev_op = None
            # 若有跨注意力后的Norm，则作为前置；否则直接使用Identity
            if hasattr(layers, "norm") or hasattr(layers, "norms"):
                # 取最后一个norm作为FFN前置（针对post-norm架构）
                prev_op = layers.norms[-1] if hasattr(layers, "norms") else layers.norm
            else:
                prev_op = nn.Identity()
            # 第一层线性（扩展维度） 
            layers.append(
                dict(
                    prev_op=prev_op,
                    layers=[layers.ffn.layers[0][0]],  # ffn.layers[0] 是Sequential，第0项是第一线性层
                    inp=input_feat["ffn.layers.0.0"],
                )
            )
        if 'ffn.layers.1' in input_feat or 'ffn.layers.1.0' in input_feat:
            # UniAD标准FFN第二层：将隐层投射回embed_dims
            # 根据实际键名选择正确的模块引用
            if 'ffn.layers.1.0' in input_feat:
                # 若存在ffn.layers.1是Sequential的情况（多于2层情况）
                prev_linear = layers.ffn.layers[1][0]
                inp_key = "ffn.layers.1.0"
            else:
                prev_linear = layers.ffn.layers[0][0] if not hasattr(layers.ffn.layers, "__getitem__") else layers.ffn.layers[1]
                inp_key = "ffn.layers.1"
            layers.append(
                dict(
                    prev_op=prev_linear,
                    layers=[layers.ffn.layers[1] if inp_key == "ffn.layers.1" else layers.ffn.layers[2]],
                    inp=input_feat[inp_key],
                )
            )
        return layers
        # layers_config = []
        # # 首先，处理 Self-Attention 部分
        # # 检查是否存在 self_attn 子模块
        # if hasattr(layer, 'self_attn'):
        #     attn = layer.self_attn
        #     # (a) 若存在显式 q_proj/k_proj/v_proj (分离的线性层)
        #     if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
        #         # 获取 self-attention 输入前的 LayerNorm (若有) 作为 prev_op
        #         prev_op = getattr(layer, 'input_layernorm', None)  # UniAD若使用PreNorm
        #         if prev_op is None:
        #             # 如果是PostNorm结构，没有输入norm，则使用Identity占位 (输入特征本身)
        #             prev_op = nn.Identity()
        #         # 将Q/K/V投影线性层合并为一组
        #         layers_config.append(dict(
        #             prev_op=prev_op,
        #             layers=[attn.q_proj, attn.k_proj, attn.v_proj],
        #             inp=input_feat.get(f"self_attn.q_proj", None),  # 使用q_proj输入作为代表（3者输入相同）
        #             module2inspect=attn,
        #             kwargs=module_kwargs
        #         ))
        #         # 输出投影层 (o_proj) 单独一组 (其输入为 v_proj 的输出)
        #         if attn.v_proj.weight.shape == attn.o_proj.weight.shape:
        #             layers_config.append(dict(
        #                 prev_op=attn.v_proj,
        #                 layers=[attn.o_proj],
        #                 inp=input_feat.get(f"self_attn.o_proj", None)
        #             ))
        #     # (b) 若 Self-Attention 是多头注意力且内部融合权重（如nn.MultiheadAttention）, 无显式q_proj等
        #     elif hasattr(attn, 'attn') and isinstance(attn.attn, nn.MultiheadAttention):
        #         # 这种情况只能量化其out_proj，qkv融合的in_proj权重因缺少独立线性层无法直接量化
        #         prev_op = getattr(layer, 'input_layernorm', None)
        #         if prev_op is None:
        #             prev_op = nn.Identity()  # 无前置Norm，用恒等映射表示
        #         # out_proj 是 nn.Linear 模块
        #         out_proj = attn.attn.out_proj
        #         layers_config.append(dict(
        #             prev_op=prev_op,
        #             layers=[out_proj],
        #             inp=input_feat.get(f"self_attn.attn.out_proj", None)
        #         ))
        #     # (c) 其他特殊注意力类型（如TemporalSelfAttention自定义实现，但通常类似多头注意力）
        #     else:
        #         # 兜底: 若以上都未匹配，自注意力包含线性但命名不同，我们获取其所有Linear子层统一处理
        #         # 将所有线性层作为一组处理
        #         linear_layers = [m for m in attn.modules() if isinstance(m, nn.Linear)]
        #         if linear_layers:
        #             prev_op = getattr(layer, 'input_layernorm', None) or nn.Identity()
        #             layers_config.append(dict(
        #                 prev_op=prev_op,
        #                 layers=linear_layers,
        #                 inp=None  # 若无法精确对应输入特征键，则留空
        #             ))
        # # 其次，处理 Cross-Attention 部分（Decoder层存在）
        # if hasattr(layer, 'cross_attn'):
        #     cross = layer.cross_attn
        #     # Cross-Attention 的输入通常是上一子层Norm输出 (self_attn输出后的Norm)
        #     prev_op = None
        #     # 寻找 cross_attn 之前的norm。例如通常Decoder结构: self_attn -> norm1 -> cross_attn
        #     if hasattr(layer, 'norm1'):
        #         prev_op = layer.norm1  # self_attn后的Norm作为cross输入
        #     elif hasattr(layer, 'prenorm') and layer.prenorm and hasattr(layer, 'self_attn_norm'):
        #         prev_op = layer.self_attn_norm  # 若采用PreNorm结构
        #     if prev_op is None:
        #         prev_op = nn.Identity()
        #     # (a) 若 cross_attn 提供 q_proj/k_proj/v_proj
        #     if hasattr(cross, 'q_proj') and hasattr(cross, 'k_proj') and hasattr(cross, 'v_proj'):
        #         # 将Q/K/V投影组合处理（query来自上一层norm输出，key/value来自memory，三者各自输入可能不同，这里简化假设QKV统一考虑）
        #         layers_config.append(dict(
        #             prev_op=prev_op,
        #             layers=[cross.q_proj, cross.k_proj, cross.v_proj],
        #             inp=input_feat.get(f"cross_attn.q_proj", None),
        #             module2inspect=cross,
        #             kwargs=module_kwargs
        #         ))
        #         # Cross-Attn 输出投影 o_proj
        #         if cross.v_proj.weight.shape == cross.o_proj.weight.shape:
        #             layers_config.append(dict(
        #                 prev_op=cross.v_proj,
        #                 layers=[cross.o_proj],
        #                 inp=input_feat.get(f"cross_attn.o_proj", None)
        #             ))
        #     # (b) 若 Cross-Attention 是多尺度可变形注意力 (MultiScaleDeformableAttention)
        #     elif hasattr(cross, 'sampling_offsets') and hasattr(cross, 'attention_weights'):
        #         # 将 sampling_offsets 与 attention_weights 两个线性层组合为一组（它们输入同为query特征，来自prev_op）
        #         layers_config.append(dict(
        #             prev_op=prev_op,
        #             layers=[cross.sampling_offsets, cross.attention_weights],
        #             inp=input_feat.get(f"cross_attn.sampling_offsets", None),
        #             module2inspect=cross,
        #             kwargs=module_kwargs
        #         ))
        #         # Value投影和输出投影的处理:
        #         # 注意: value_proj 输入来自memory特征，与query分布不同；output_proj输入是注意力聚合结果。
        #         # 这里我们将value_proj单独量化，再量化output_proj，近似假设value_proj的量化不受输入分布差异太大影响。
        #         if hasattr(cross, 'value_proj') and hasattr(cross, 'output_proj'):
        #             # 为简化，使用 sampling_offsets 前的 prev_op (同样的norm) 近似作为 value_proj 的 prev_op
        #             layers_config.append(dict(
        #                 prev_op=prev_op,
        #                 layers=[cross.value_proj],
        #                 inp=input_feat.get(f"cross_attn.value_proj", None)
        #             ))
        #             # 将 output_proj 视为接在 value_proj 之后
        #             if cross.value_proj.weight.shape == cross.output_proj.weight.shape:
        #                 layers_config.append(dict(
        #                     prev_op=cross.value_proj,
        #                     layers=[cross.output_proj],
        #                     inp=input_feat.get(f"cross_attn.output_proj", None)
        #                 ))
        #     # (c) 其他类型（如 MultiheadAttention）
        #     else:
        #         # 若cross_attn是类似MultiheadAttention的封装
        #         if hasattr(cross, 'attn') and isinstance(cross.attn, nn.MultiheadAttention):
        #             out_proj = cross.attn.out_proj
        #             layers_config.append(dict(
        #                 prev_op=prev_op,
        #                 layers=[out_proj],
        #                 inp=input_feat.get(f"cross_attn.attn.out_proj", None)
        #             ))
        #         else:
        #             # 兜底: 直接量化cross_attn模块中全部Linear子层
        #             linear_layers = [m for m in cross.modules() if isinstance(m, nn.Linear)]
        #             if linear_layers:
        #                 layers_config.append(dict(
        #                     prev_op=prev_op,
        #                     layers=linear_layers,
        #                     inp=None
        #                 ))
        # # 最后，处理前馈网络 (Feed-Forward Network, FFN) 部分
        # # UniAD的Transformer层通常有 FFN, 可能作为 layer.ffn (mmcv FFN模块) 或两个Linear (如 layer.linear1, layer.linear2)
        # # 典型结构: LN(前) -> Linear1 -> 激活 -> Dropout -> Linear2 -> Dropout -> 残差连接 -> LN(后)
        # # 为量化，我们将Linear1和Linear2分别处理:
        # # Linear1使用其输入(即上一子层输出经过norm/激活后的特征)计算scale; Linear2使用Linear1输出(经过激活)的特征计算scale。
        # ffn_first_linear = None
        # ffn_second_linear = None
        # # 检查常见属性名称
        # if hasattr(layer, 'ffn'):
        #     # mmcv的FFN模块内部可能用Sequential存储层
        #     # 尝试获取其中的线性层
        #     for name, subm in layer.ffn.named_modules():
        #         if isinstance(subm, nn.Linear):
        #             if ffn_first_linear is None:
        #                 ffn_first_linear = subm
        #             else:
        #                 # 将最后一个Linear赋给second_linear (FFN通常只有两层Linear)
        #                 ffn_second_linear = subm
        #     # 如果通过遍历未正确获取，可直接取 Sequential的第一个和最后一个Linear
        #     if ffn_first_linear is None or ffn_second_linear is None:
        #         if hasattr(layer.ffn, 'layers') and isinstance(layer.ffn.layers, nn.Sequential):
        #             seq_layers = [m for m in layer.ffn.layers if isinstance(m, nn.Linear)]
        #             if seq_layers:
        #                 ffn_first_linear = seq_layers[0]
        #                 ffn_second_linear = seq_layers[-1] if len(seq_layers) > 1 else None
        # else:
        #     # 有些实现可能直接在layer中有 linear1/linear2 属性
        #     if hasattr(layer, 'linear1'):
        #         ffn_first_linear = layer.linear1
        #     if hasattr(layer, 'linear2'):
        #         ffn_second_linear = layer.linear2
        # # 添加FFN层的量化配置
        # if ffn_first_linear is not None:
        #     # FFN第一层 Linear
        #     # Prev_op 选取FFN输入之前的norm。如果是Decoder，通常 cross_attn后有 norm2，再进入FFN
        #     prev_op = None
        #     if hasattr(layer, 'norm2'):
        #         prev_op = layer.norm2  # 使用cross_attn后的norm输出作为FFN输入
        #     elif hasattr(layer, 'ffn_norm'):
        #         prev_op = layer.ffn_norm  # 有的实现中专门命名FFN前的Norm
        #     if prev_op is None:
        #         prev_op = nn.Identity()
        #     layers_config.append(dict(
        #         prev_op=prev_op,
        #         layers=[ffn_first_linear],
        #         inp=input_feat.get(f"{ffn_first_linear.__class__.__name__}", None)  # 若input_feat有记录Linear名称可填
        #     ))
        # if ffn_second_linear is not None:
        #     # FFN第二层 Linear，以第一层Linear输出作为前置op（近似，忽略激活对分布的影响）
        #     prev_op = ffn_first_linear if ffn_first_linear is not None else nn.Identity()
        #     layers_config.append(dict(
        #         prev_op=prev_op,
        #         layers=[ffn_second_linear],
        #         inp=input_feat.get(f"{ffn_second_linear.__class__.__name__}", None)
        #     ))
        # return layers_config
