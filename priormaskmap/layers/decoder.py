import torch
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from priormaskmap.layers.bev.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32,
)
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner.base_module import BaseModule
import warnings
from mmcv.cnn import xavier_init, constant_init
import torch.nn as nn
import copy
import math


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    return pos


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PolylineDecoder(TransformerLayerSequence):
    def __init__(
        self,
        *args,
        return_intermediate=False,
        query_pos_embedding="none",
        num_pts_per_vec=20,
        **kwargs,
    ):
        super(PolylineDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.query_pos_embedding = query_pos_embedding
        self.num_pts_per_vec = num_pts_per_vec
        if query_pos_embedding == "instance":

            def _get_clones(module, N):
                return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

            pt_pos_query_proj = nn.Linear(self.embed_dims, self.embed_dims)
            self.pt_pos_query_projs = _get_clones(pt_pos_query_proj, self.num_layers)

    def init_weights(self):
        if self.query_pos_embedding == "instance":
            for m in self.pt_pos_query_projs:
                xavier_init(m, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        bs = reference_points.shape[0]
        for lid, layer in enumerate(self.layers):
            if self.query_pos_embedding == "instance":
                reference_points_reshape = reference_points.view(
                    bs, -1, self.num_pts_per_vec, 2
                )
                reference_points_reshape = reference_points_reshape.view(bs, -1, 2)
                query_sine_embed = gen_sineembed_for_position(reference_points_reshape)
                query_sine_embed = query_sine_embed.view(
                    bs, -1, self.num_pts_per_vec, self.embed_dims
                )
                point_query_pos = self.pt_pos_query_projs[lid](query_sine_embed)
                query_pos_lid = None
                reference_points_input = reference_points_reshape[..., :2].unsqueeze(2)
            else:
                point_query_pos = None
                reference_points_input = reference_points[..., :2].unsqueeze(2)
                query_pos_lid = query_pos
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos_lid,
                pt_query_pos=point_query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
            )
        return (output, reference_points)


@ATTENTION.register_module()
class PriorInstancePointAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        num_pts_per_vec=20,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        def _is_power_of_2(n):
            return n & n - 1 == 0 and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in MultiScaleDeformAttention to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation."
            )
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims * num_pts_per_vec, embed_dims * num_pts_per_vec // 2),
            nn.ReLU(),
            nn.Linear(embed_dims * num_pts_per_vec // 2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        pt_query_pos=None,
        flag="decoder",
        **kwargs,
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        if pt_query_pos is not None:
            bs, num_iq, num_pt, _ = pt_query_pos.shape
            query = query.unsqueeze(2) + pt_query_pos
            query = query.reshape(bs, num_iq * num_pt, -1)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = reference_points[
                :, :, None, :, None, :2
            ] + sampling_offsets / (
                offset_normalizer[None, None, None, :, None, :]
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        output = output.view(bs, num_iq, num_pt, -1)
        output = output.flatten(2)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


@TRANSFORMER_LAYER.register_module()
class PriorDecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        num_vec=50,
        num_pts_per_vec=20,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(PriorDecoupledDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in {self.__class__.__name__} "
            )
        num_vec = kwargs["num_vec"]
        num_pts_per_vec = kwargs["num_pts_per_vec"]
        for layer in self.operation_order:
            if layer == "self_attn":
                if attn_index == 0:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(
                        num_vec, num_pts_per_vec, n_batch, n_dim
                    ).flatten(1, 2)
                    if query_pos is not None:
                        query_pos = query_pos.view(
                            num_vec, num_pts_per_vec, n_batch, n_dim
                        ).flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=kwargs["self_attn_mask"],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    query = query.view(
                        num_vec, num_pts_per_vec, n_batch, n_dim
                    ).flatten(0, 1)
                    if query_pos is not None:
                        query_pos = query_pos.view(
                            num_vec, num_pts_per_vec, n_batch, n_dim
                        ).flatten(0, 1)
                    attn_index += 1
                    identity = query
                else:
                    n_pts, n_batch, n_dim = query.shape
                    query = (
                        query.view(num_vec, num_pts_per_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(1, 2)
                    )
                    if query_pos is not None:
                        query_pos = (
                            query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim)
                            .permute(1, 0, 2, 3)
                            .contiguous()
                            .flatten(1, 2)
                        )
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    query = (
                        query.view(num_pts_per_vec, num_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(0, 1)
                    )
                    if query_pos is not None:
                        query_pos = (
                            query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim)
                            .permute(1, 0, 2, 3)
                            .contiguous()
                            .flatten(0, 1)
                        )
                    attn_index += 1
                    identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query
