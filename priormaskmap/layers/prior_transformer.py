import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torchvision.transforms.functional import rotate
from priormaskmap.layers.bev.temporal_self_attention import PriorTemporalSelfAttention
from priormaskmap.layers.bev.spatial_cross_attention import PriorMSDeformableAttention3D
from priormaskmap.layers.bev.decoder import PriorCustomMSDeformableAttention
from priormaskmap.layers.decoder import PriorInstancePointAttention
from priormaskmap.layers.height_kernel_attention import (
    PriorHeightKernelAttention,
    PriorMSDeformableAttentionKernel,
)
from priormaskmap.layers.builder import build_fuser


@TRANSFORMER.register_module()
class PriorFusionTransformer(BaseModule):
    """Fuse BEV features and decode prior-conditioned map queries."""

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
        two_stage_num_proposals=300,
        fuser=None,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        modality="vision",
        feat_down_sample_indice=-1,
        query_pos_embedding="none",
        map_encoder=None,
        enable_expert_fusion=True,
        vector_query_weight=0.5,
        fusion_type="uncertainty_guided_attention",
        **kwargs,
    ):
        super(PriorFusionTransformer, self).__init__(**kwargs)
        if modality == "fusion":
            self.fuser = build_fuser(fuser)
        self.use_attn_bev = "PriorAttentionEncoder" in encoder["type"]
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        if map_encoder is not None:
            self.map_encoder = build_transformer_layer_sequence(map_encoder)
        else:
            self.map_encoder = None
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg = z_cfg
        self.query_pos_embedding = query_pos_embedding
        self.num_pts_per_vec = self.decoder.num_pts_per_vec
        self.rotate_center = rotate_center
        self.feat_down_sample_indice = feat_down_sample_indice
        self.enable_expert_fusion = enable_expert_fusion
        self.vector_query_weight = vector_query_weight
        self.fusion_type = fusion_type
        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        if (
            self.query_pos_embedding == "instance"
            or self.query_pos_embedding == "map_instance_pos"
        ):
            self.reference_points = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.num_pts_per_vec * 2),
            )
        elif self.query_pos_embedding == "map_pts":
            self.reference_points = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, 2),
            )
        elif self.query_pos_embedding == "map_instance":
            self.reference_points = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims * 2),
                nn.ReLU(),
                nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
                nn.ReLU(),
                nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
                nn.ReLU(),
                nn.Linear(self.embed_dims * 2, self.num_pts_per_vec * 2),
            )
        else:
            self.reference_points = (
                nn.Linear(self.embed_dims, 2)
                if not self.z_cfg["gt_z_flag"]
                else nn.Linear(self.embed_dims, 3)
            )
        if "map" in self.query_pos_embedding:
            self.map_feats_conv = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims + 3, self.embed_dims, 3, padding=1, bias=False
                ),
                nn.BatchNorm2d(self.embed_dims),
                nn.ReLU(True),
            )
            self.uncertainty_gate = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims + 3, self.embed_dims // 2, 3, padding=1, bias=False
                ),
                nn.BatchNorm2d(self.embed_dims // 2),
                nn.ReLU(True),
                nn.Conv2d(self.embed_dims // 2, 1, 1, padding=0, bias=True),
                nn.Sigmoid(),
            )
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))
        if self.enable_expert_fusion:
            self._init_expert_fusion_layers()

    def _init_expert_fusion_layers(self):
        self.vector_query_attention = nn.MultiheadAttention(
            self.embed_dims, num_heads=8, batch_first=True, dropout=0.05
        )
        self.fusion_norm = nn.LayerNorm(self.embed_dims)
        if self.fusion_type == "gate":
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.embed_dims * 2, self.embed_dims), nn.Sigmoid()
            )
        elif self.fusion_type == "weighted":
            self.fusion_weight = nn.Parameter(
                torch.tensor(self.vector_query_weight, dtype=torch.float64)
            )
        if self.fusion_type in [
            "uncertainty_guided_attention",
            "adaptive_dual_attention",
        ]:
            self.uncertainty_modulator = nn.Sequential(
                nn.Linear(1, self.embed_dims // 4),
                nn.ReLU(),
                nn.Linear(self.embed_dims // 4, 1),
                nn.Sigmoid(),
            )
        self.uncertainty_proj = nn.Linear(1, self.embed_dims)

    def _generate_confidence_dilated_mask(self, local_map, uncertainty_map=None):
        bs, _, bev_h, bev_w = local_map.shape
        device = local_map.device
        dtype = local_map.dtype
        local_map_confidence, _ = torch.max(local_map, dim=1)
        fusion_weights = torch.zeros(bs, 1, bev_h, bev_w, device=device, dtype=dtype)

        def apply_diffusion(
            region_mask: torch.Tensor, kernel_size: int
        ) -> torch.Tensor:
            region_mask = region_mask.to(dtype=dtype)
            if kernel_size <= 1:
                return region_mask
            kernel = torch.ones(kernel_size, kernel_size, device=device, dtype=dtype)
            kernel = kernel / kernel.sum()
            padding = kernel_size // 2
            dilated = F.conv2d(
                region_mask.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=padding,
            )[0, 0]
            target_h, target_w = region_mask.shape
            if dilated.shape[0] > target_h:
                start = (dilated.shape[0] - target_h) // 2
                dilated = dilated[start : start + target_h, :]
            if dilated.shape[1] > target_w:
                start = (dilated.shape[1] - target_w) // 2
                dilated = dilated[:, start : start + target_w]
            if dilated.shape[0] < target_h or dilated.shape[1] < target_w:
                pad_bottom = target_h - dilated.shape[0]
                pad_right = target_w - dilated.shape[1]
                dilated = F.pad(dilated, (0, pad_right, 0, pad_bottom), value=0)
            return dilated

        uncertainty_prob = F.softmax(uncertainty_map, dim=1)
        prob_correct = uncertainty_prob[:, 0, :, :]
        prob_error = uncertainty_prob[:, 1, :, :]
        weight_map = {1: 1.0, 2: 0.7, 3: 0.2}
        kernel_size_map = {1: 2, 2: 5, 3: 7}
        for b in range(bs):
            batch_weights = torch.zeros_like(local_map_confidence[b])
            has_map_mask = prob_correct[b] + prob_error[b] > 0.5
            if has_map_mask.any():
                correct_confidence = prob_correct[b][has_map_mask]
                high_conf_mask = correct_confidence > 0.7
                med_conf_mask = (correct_confidence >= 0.3) & (
                    correct_confidence <= 0.7
                )
                low_conf_mask = (correct_confidence > 0) & (correct_confidence <= 0.3)
                has_map_indices = torch.where(has_map_mask)
                diffusion_sizes = torch.zeros_like(batch_weights, dtype=torch.int64)
                if high_conf_mask.any():
                    high_indices = (
                        has_map_indices[0][high_conf_mask],
                        has_map_indices[1][high_conf_mask],
                    )
                    diffusion_sizes[high_indices] = 1
                if med_conf_mask.any():
                    med_indices = (
                        has_map_indices[0][med_conf_mask],
                        has_map_indices[1][med_conf_mask],
                    )
                    diffusion_sizes[med_indices] = 2
                if low_conf_mask.any():
                    low_indices = (
                        has_map_indices[0][low_conf_mask],
                        has_map_indices[1][low_conf_mask],
                    )
                    diffusion_sizes[low_indices] = 3
                for level, kernel_size in kernel_size_map.items():
                    region_mask = diffusion_sizes == level
                    if region_mask.any():
                        dilated = apply_diffusion(region_mask.float(), kernel_size)
                        batch_weights = torch.maximum(
                            batch_weights, dilated * weight_map[level]
                        )
            fusion_weights[b, 0] = batch_weights
        return fusion_weights

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, PriorHeightKernelAttention):
                m.init_weight()
            elif (
                isinstance(m, PriorMSDeformableAttention3D)
                or isinstance(m, PriorTemporalSelfAttention)
                or isinstance(m, PriorCustomMSDeformableAttention)
                or isinstance(m, PriorInstancePointAttention)
                or isinstance(m, PriorMSDeformableAttentionKernel)
            ):
                m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        self.decoder.init_weights()
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        xavier_init(self.can_bus_mlp, distribution="uniform", bias=0.0)
        if "map" in self.query_pos_embedding:
            xavier_init(self.map_feats_conv, distribution="uniform", bias=0.0)
            xavier_init(self.uncertainty_gate, distribution="uniform", bias=0.0)
        if self.enable_expert_fusion:
            if hasattr(self, "fusion_norm"):
                nn.init.constant_(self.fusion_norm.weight, 1.0)
                nn.init.constant_(self.fusion_norm.bias, 0.0)
            if hasattr(self, "fusion_gate"):
                xavier_init(self.fusion_gate, distribution="uniform", bias=0.0)
            if hasattr(self, "uncertainty_modulator"):
                xavier_init(
                    self.uncertainty_modulator, distribution="uniform", bias=0.0
                )
            if hasattr(self, "uncertainty_proj"):
                xavier_init(self.uncertainty_proj, distribution="uniform", bias=0.0)

    def attn_bev_encode(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        delta_x = np.array([each["can_bus"][0] for each in kwargs["img_metas"]])
        delta_y = np.array([each["can_bus"][1] for each in kwargs["img_metas"]])
        ego_angle = np.array(
            [each["can_bus"][-2] / np.pi * 180 for each in kwargs["img_metas"]]
        )
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = (
            translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        )
        shift_x = (
            translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        )
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs["img_metas"][i]["can_bus"][-1]
                    tmp_prev_bev = (
                        prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    )
                    tmp_prev_bev = rotate(
                        tmp_prev_bev, rotation_angle, center=self.rotate_center
                    )
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1
                    )
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
        can_bus = bev_queries.new_tensor(
            [each["can_bus"] for each in kwargs["img_metas"]]
        )
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        ret_dict = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            mlvl_feats=mlvl_feats,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )
        return ret_dict

    def lss_bev_encode(self, mlvl_feats, prev_bev=None, **kwargs):
        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs["img_metas"]
        encoder_outputdict = self.encoder(images, img_metas)
        bev_embed = encoder_outputdict["bev"]
        depth = encoder_outputdict["depth"]
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()
        ret_dict = dict(bev=bev_embed, depth=depth)
        return ret_dict

    def get_bev_features(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        if self.use_attn_bev:
            ret_dict = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs,
            )
            bev_embed = ret_dict["bev"]
            depth = ret_dict["depth"]
        else:
            ret_dict = self.lss_bev_encode(mlvl_feats, prev_bev=prev_bev, **kwargs)
            bev_embed = ret_dict["bev"]
            depth = ret_dict["depth"]
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = (
                bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            )
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()
            lidar_feat = nn.functional.interpolate(
                lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False
            )
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev
        ret_dict = dict(bev=bev_embed, depth=depth)
        return ret_dict

    def format_feats(self, mlvl_feats):
        bs = mlvl_feats[0].size(0)
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)).to(torch.int64),
                spatial_shapes.prod(1).cumsum(0)[:-1].to(torch.int64),
            )
        )
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        return (feat_flatten, spatial_shapes, level_start_index)

    def forward(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        local_map=None,
        uncertainty_map=None,
        reshape=True,
        **kwargs,
    ):
        ouput_dic = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            **kwargs,
        )
        bev_embed = ouput_dic["bev"]
        depth = ouput_dic["depth"]
        bs = mlvl_feats[0].size(0)
        if self.map_encoder is not None:
            bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
            feat_flatten = []
            spatial_shapes = []
            for lvl, feat in enumerate(mlvl_feats):
                bs, num_cam, c, h, w = feat.shape
                spatial_shape = (h, w)
                feat = feat.flatten(3).permute(1, 0, 3, 2)
                if self.use_cams_embeds:
                    feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
                feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(
                    feat.dtype
                )
                spatial_shapes.append(spatial_shape)
                feat_flatten.append(feat)
            feat_flatten = torch.cat(feat_flatten, 2)
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=bev_pos.device
            )
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
            )
            feat_flatten = feat_flatten.permute(0, 2, 1, 3)
            bev_embed_map = self.map_encoder(
                bev_queries,
                feat_flatten,
                feat_flatten,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs,
            )["bev"]
            bev_embed_map = (
                bev_embed_map.permute(0, 2, 1).view(bs, -1, bev_h, bev_w).contiguous()
            )
            fusion_weights = self._generate_confidence_dilated_mask(
                local_map, uncertainty_map
            )
            bev_embed = (
                bev_embed.permute(0, 2, 1).view(bs, -1, bev_h, bev_w).contiguous()
            )
            fused_embed = fusion_weights * bev_embed_map + bev_embed
            bev_embed = fused_embed
        if local_map is not None:
            if reshape:
                local_map_reshape = (
                    local_map.permute(1, 0, 2)
                    .contiguous()
                    .view(bs, bev_h, bev_w, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            else:
                local_map_reshape = local_map
            if uncertainty_map is not None:
                confidence_map = torch.sigmoid(uncertainty_map)
                gate_input = torch.cat((bev_embed, local_map_reshape), dim=1)
                gate_weights = self.uncertainty_gate(gate_input)
                final_weights = confidence_map * gate_weights
                local_map_weighted = local_map_reshape * final_weights
                bev_fuse = torch.cat((bev_embed, local_map_weighted), dim=1)
                bev_embed = self.map_feats_conv(bev_fuse)
            else:
                bev_fuse = torch.cat((bev_embed, local_map_reshape), dim=1)
                bev_embed = self.map_feats_conv(bev_fuse)
            bev_embed = bev_embed.flatten(2).permute(0, 2, 1).contiguous()
        if (
            self.query_pos_embedding == "map_pts"
            or self.query_pos_embedding == "map_instance"
        ):
            query = object_query_embed
            reference_points = self.reference_points(query).sigmoid()
            init_reference_out = reference_points
            query_pos = None
        elif self.query_pos_embedding == "instance":
            query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
            reference_points = self.reference_points(query_pos).sigmoid()
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
            query_pos = None
        elif self.query_pos_embedding == "map_instance_pos":
            query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=-1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points
            query_pos = None
        else:
            query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)
            reference_points = reference_points.sigmoid()
            init_reference_out = reference_points
            query_pos = query_pos.permute(1, 0, 2)
        query = query.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        feat_flatten, feat_spatial_shapes, feat_level_start_index = self.format_feats(
            mlvl_feats
        )
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor(
                [[bev_h, bev_w]], device=query.device, dtype=torch.int64
            ),
            level_start_index=torch.tensor([0], device=query.device, dtype=torch.int64),
            mlvl_feats=mlvl_feats,
            feat_flatten=feat_flatten,
            feat_spatial_shapes=feat_spatial_shapes,
            feat_level_start_index=feat_level_start_index,
            **kwargs,
        )
        inter_references_out = inter_references
        return (
            bev_embed,
            depth,
            inter_states,
            init_reference_out,
            inter_references_out,
        )
