import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import multi_apply, reduce_mean
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.cnn.bricks.transformer import MultiheadAttention
from priormaskmap.models.unet import PriorUNet
from priormaskmap.models.vector_encoder import PriorVectorEncoder


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, ignore_index=-100):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        self.weights = weights / weights.sum() * len(weights)

    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)
        ce_loss = F.cross_entropy(
            inputs, targets, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        batch_weights = weights[targets]
        cb_loss = batch_weights * (1 - pt) ** self.gamma * ce_loss
        return cb_loss.mean()


def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    new_pts[..., 2:3] = pts[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return new_pts


def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    patch_z = pc_range[5] - pc_range[2]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    new_pts[..., 2:3] = pts[..., 2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h, patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


@HEADS.register_module()
class PriorMaskMapHead(DETRHead):
    """Build prior queries and predict classes and polyline coordinates."""

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        num_vec_one2one=50,
        num_vec_one2many=0,
        k_one2many=0,
        lambda_one2many=1,
        num_pts_per_vec=2,
        num_pts_per_gt_vec=2,
        query_embed_type="all_pts",
        map_query_scale=2,
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        aux_seg=dict(
            use_aux_seg=False,
            bev_seg=False,
            pv_seg=False,
            seg_classes=1,
            feat_down_sample=32,
        ),
        z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
        enable_error_detection=False,
        error_detection_loss_weight=1.0,
        uncertainty_loss_type="focal",
        class_weights=[5.0, 5.0, 1.0],
        focal_alpha=1.0,
        focal_gamma=2.0,
        cb_beta=0.9999,
        cb_gamma=2.0,
        vector_embed_dim=128,
        vector_num_heads=8,
        vector_M=3,
        vector_N=2,
        vector_num_types=3,
        loss_pts=dict(type="ChamferDistance", loss_src_weight=1.0, loss_dst_weight=1.0),
        loss_seg=dict(type="PriorSimpleLoss", pos_weight=2.13, loss_weight=1.0),
        loss_pv_seg=dict(type="PriorSimpleLoss", pos_weight=2.13, loss_weight=1.0),
        loss_dir=dict(type="PriorPtsDirCosLoss", loss_weight=2.0),
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = 2 if not z_cfg["pred_z_flag"] else 3
        else:
            self.code_size = 2
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.query_embed_type = query_embed_type
        if "pts" not in self.query_embed_type:
            self.code_size = self.code_size * num_pts_per_vec
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_vec = num_vec_one2one + num_vec_one2many
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.aux_seg = aux_seg
        self.z_cfg = z_cfg
        self.enable_error_detection = True
        self.error_detection_loss_weight = 1.0
        self.uncertainty_loss_type = uncertainty_loss_type
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cb_beta = cb_beta
        self.cb_gamma = cb_gamma
        self.vector_embed_dim = vector_embed_dim
        self.vector_num_heads = vector_num_heads
        self.vector_M = vector_M
        self.vector_N = vector_N
        self.vector_num_types = vector_num_types
        super(PriorMaskMapHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.vector_uve = PriorVectorEncoder(
            point_dim=2,
            embed_dim=vector_embed_dim,
            num_heads=vector_num_heads,
            M=vector_M,
            N=vector_N,
            num_types=vector_num_types,
            target_embed_dim=self.embed_dims,
        )
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.loss_seg = build_loss(loss_seg)
        self.loss_pv_seg = build_loss(loss_pv_seg)
        if self.enable_error_detection:
            self.error_detection_unet = PriorUNet(
                num_classes=3,
                in_channels=self.embed_dims + 1,
                depth=4,
                start_filts=64,
                up_mode="transpose",
                merge_mode="concat",
            )
            if self.uncertainty_loss_type == "ce":
                self.loss_error_detection = nn.CrossEntropyLoss()
            elif self.uncertainty_loss_type == "weighted_ce":
                weights = torch.tensor(self.class_weights, dtype=torch.float32)
                self.loss_error_detection = nn.CrossEntropyLoss(weight=weights)
            elif self.uncertainty_loss_type == "focal":
                self.loss_error_detection = FocalLoss(
                    alpha=self.focal_alpha, gamma=self.focal_gamma
                )
            elif self.uncertainty_loss_type == "cb":
                samples_per_class = torch.tensor([0.01, 0.01, 0.98])
                self.loss_error_detection = ClassBalancedLoss(
                    samples_per_class=samples_per_class,
                    beta=self.cb_beta,
                    gamma=self.cb_gamma,
                )
        self.vector_query_proj = nn.Linear(vector_embed_dim, self.embed_dims)
        self.map_query_scale = map_query_scale
        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)
        reg_branch = []
        if "pts" not in self.query_embed_type:
            reg_branch.append(Linear(self.embed_dims, self.embed_dims * 2))
            reg_branch.append(nn.ReLU())
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims * 2, self.embed_dims * 2))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims * 2, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)
        else:
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (
            self.transformer.decoder.num_layers + 1
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        if self.aux_seg["use_aux_seg"]:
            if self.aux_seg["bev_seg"]:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(
                        self.embed_dims,
                        self.embed_dims,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.embed_dims,
                        self.aux_seg["seg_classes"],
                        kernel_size=1,
                        padding=0,
                    ),
                )
            if self.aux_seg["pv_seg"]:
                self.pv_seg_head = nn.Sequential(
                    nn.Conv2d(
                        self.embed_dims,
                        self.embed_dims,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.embed_dims,
                        self.aux_seg["seg_classes"],
                        kernel_size=1,
                        padding=0,
                    ),
                )
        if not self.as_two_stage:
            if (
                "PriorAttentionEncoder" in self.bev_encoder_type
                or self.transformer.map_encoder is not None
            ):
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims
                )
            else:
                self.bev_embedding = None
            if self.query_embed_type == "all_pts":
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            elif self.query_embed_type == "instance_pts":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(
                    self.num_vec, self.embed_dims * 2
                )
                self.pts_embedding = nn.Embedding(
                    self.num_pts_per_vec, self.embed_dims * 2
                )
            elif self.query_embed_type == "instance":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(
                    self.num_vec, self.embed_dims * 2
                )
                self.pts_embedding = None
            elif self.query_embed_type == "map_pts":
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
                self.label_encoder = nn.Linear(3, self.embed_dims)
                self.query_attention = MultiheadAttention(
                    self.embed_dims, num_heads=4, batch_first=True, dropout=0.0
                )
            elif self.query_embed_type == "map_instance":
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)
                self.pts_embedding = None
                self.label_encoder = nn.Linear(3, self.embed_dims)
                self.query_attention = MultiheadAttention(
                    self.embed_dims, num_heads=4, batch_first=True, dropout=0.1
                )
            elif self.query_embed_type == "map_instance_pos":
                self.instance_embedding = nn.Embedding(
                    self.num_vec, self.embed_dims * 2
                )
                self.pts_embedding = None
                self.label_encoder = nn.Linear(3, self.embed_dims)
                self.query_attention = MultiheadAttention(
                    self.embed_dims * 2, num_heads=4, batch_first=True, dropout=0.1
                )

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if "map" in self.query_embed_type:
            xavier_init(self.label_encoder, distribution="uniform", bias=0.0)

    def _detect_map_errors(self, bev_embed, local_map, bs):
        if bev_embed.dim() == 3:
            bev_embed = (
                bev_embed.permute(1, 2, 0)
                .contiguous()
                .view(bs, self.embed_dims, self.bev_h, self.bev_w)
            )
        elif bev_embed.dim() == 2:
            bev_embed = (
                bev_embed.unsqueeze(0)
                .expand(bs, -1, -1)
                .permute(0, 2, 1)
                .view(bs, self.embed_dims, self.bev_h, self.bev_w)
            )
        local_map_reshaped = local_map
        local_map_single_channel, _ = torch.max(local_map_reshaped, dim=1, keepdim=True)
        local_map_filtered = torch.where(
            local_map_single_channel > 0.8,
            local_map_single_channel,
            torch.zeros_like(local_map_single_channel),
        )
        local_map_reshaped = local_map_filtered
        unet_input = torch.cat([bev_embed, local_map_reshaped], dim=1)
        uncertainty_pred = self.error_detection_unet(unet_input)
        return uncertainty_pred

    def _generate_filtered_map_valid(self, local_map_input, uncertain_pred):
        local_map_valid, _ = torch.max(local_map_input, dim=1)
        uncertain_pred = F.interpolate(
            uncertain_pred,
            size=(local_map_valid.shape[1], local_map_valid.shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        uncertain_pred = torch.softmax(uncertain_pred, dim=1)
        have_map_mask = uncertain_pred[:, 2, :, :] >= 0.4
        valid_mask = uncertain_pred[:, 0, :, :] > uncertain_pred[:, 1, :, :]
        local_map_valid = local_map_valid * valid_mask.float() * have_map_mask.float()
        return local_map_valid

    def _generate_uncertainty_weighted_map(self, local_map_input, uncertainty_pred):
        local_map_valid, _ = torch.max(local_map_input, dim=-1)
        uncertainty_resized = F.interpolate(
            uncertainty_pred,
            size=(local_map_input.shape[1], local_map_input.shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        uncertainty_prob = F.softmax(uncertainty_resized, dim=1)
        prob_correct = uncertainty_prob[:, 0, :, :]
        prob_error = uncertainty_prob[:, 1, :, :]
        prob_no_map = uncertainty_prob[:, 2, :, :]
        confidence_map = prob_correct * 1.0 + prob_error * 0.0 + prob_no_map * 0.0
        local_map_valid = local_map_valid * confidence_map
        return (local_map_valid, confidence_map)

    @force_fp32(apply_to=("mlvl_feats", "local_map"))
    def forward(
        self,
        mlvl_feats,
        lidar_feat,
        img_metas,
        local_map=None,
        only_bev=False,
        vector_maps_gt=None,
        vector_maps_modified=None,
    ):
        if self.training:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        elif self.query_embed_type == "instance":
            object_query_embeds = self.instance_embedding.weight[0:num_vec].to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)
            bev_mask = torch.zeros(
                (bs, self.bev_h, self.bev_w), device=bev_queries.device
            ).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None
        if self.query_embed_type == "map_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)
            query = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
            uncertainty_pred = None
            if self.enable_error_detection and local_map is not None:
                temp_outputs = self.transformer.get_bev_features(
                    mlvl_feats,
                    lidar_feat,
                    bev_queries,
                    self.bev_h,
                    self.bev_w,
                    grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                    bev_pos=bev_pos,
                    img_metas=img_metas,
                    local_map=None,
                )["bev"]
                temp_bev_embed = (
                    temp_outputs
                    if isinstance(temp_outputs, torch.Tensor)
                    else temp_outputs[0]
                )
                uncertainty_pred = self._detect_map_errors(
                    temp_bev_embed, local_map, bs
                )
            local_map_input = F.interpolate(
                local_map,
                [
                    self.bev_h // self.map_query_scale,
                    self.bev_w // self.map_query_scale,
                ],
            )
            local_map_valid = self._generate_filtered_map_valid(
                local_map_input, uncertainty_pred
            )
            local_map_pos = F.interpolate(
                bev_pos,
                [
                    self.bev_h // self.map_query_scale,
                    self.bev_w // self.map_query_scale,
                ],
            )
            valid_index = local_map_valid > 0.5
            bs = local_map_pos.shape[0]
            pos_reshaped = local_map_pos.reshape(bs, local_map_pos.shape[1], -1)
            label_reshaped = local_map_input.reshape(bs, local_map_input.shape[1], -1)
            valid_index_reshaped = valid_index.reshape(bs, -1)
            pos_results = []
            label_results = []
            for i in range(bs):
                single_mask = valid_index_reshaped[i]
                pos_results.append(pos_reshaped[i][:, single_mask])
                label_results.append(label_reshaped[i][:, single_mask])
            map_pos_embed_padded = torch.nn.utils.rnn.pad_sequence(
                [t.transpose(0, 1) for t in pos_results],
                batch_first=True,
                padding_value=0.0,
            )
            map_label_padded = torch.nn.utils.rnn.pad_sequence(
                [t.transpose(0, 1) for t in label_results],
                batch_first=True,
                padding_value=0.0,
            )
            map_label_embed = self.label_encoder(map_label_padded.squeeze(-1))
            pts_map_query = map_pos_embed_padded + map_label_embed
            if vector_maps_modified is not None:
                if uncertainty_pred is not None:
                    confidence_scores = self._extract_uncertainty_confidence_for_points(
                        vector_maps_modified, uncertainty_pred
                    )
                    confidence_threshold = 0.01
                    point_mask = confidence_scores > confidence_threshold
                    coordinate_mask = ~point_mask.unsqueeze(-1).expand(-1, -1, -1, 256)
                vector_point_features = self.vector_uve(vector_maps_modified)
                vector_point_features = vector_point_features * coordinate_mask
                vector_features_flat = vector_point_features.view(
                    bs, -1, self.embed_dims
                )
                vector_features_final = vector_features_flat
                query_batch = query.unsqueeze(0).expand(bs, -1, -1)
                fused_query = torch.cat([vector_features_final, pts_map_query], dim=1)
                fused_query = self.query_attention(
                    query_batch, key=fused_query, value=fused_query
                )
                object_query_embeds = fused_query
            else:
                object_query_embeds = query.unsqueeze(0).expand(bs, -1, -1)
        elif self.query_embed_type == "map_instance":
            local_map_input = (
                local_map.permute(1, 0, 2)
                .contiguous()
                .view(bs, self.bev_h, self.bev_w, -1)
                .permute(0, 3, 1, 2)
            )
            local_map_input = F.interpolate(
                local_map_input,
                [
                    self.bev_h // self.map_query_scale,
                    self.bev_w // self.map_query_scale,
                ],
            ).permute(0, 2, 3, 1)
            local_map_pos = F.interpolate(
                bev_pos,
                [
                    self.bev_h // self.map_query_scale,
                    self.bev_w // self.map_query_scale,
                ],
            ).permute(0, 2, 3, 1)
            local_map_valid, _ = torch.max(local_map_input, dim=-1)
            query_embeds_list = []
            for i in range(bs):
                valid_index = local_map_valid[i] > 0.8
                if torch.equal(
                    valid_index,
                    torch.zeros(
                        valid_index.shape, dtype=torch.bool, device=valid_index.device
                    ),
                ):
                    query_embeds_list.append(
                        self.instance_embedding.weight[0:num_vec].to(dtype)
                    )
                else:
                    map_pos_embed = local_map_pos[i][valid_index]
                    map_label = local_map_input[i][valid_index]
                    map_label_embed = self.label_encoder(map_label)
                    map_query = map_pos_embed + map_label_embed
                    query_embeds = self.instance_embedding.weight[0:num_vec].to(dtype)
                    query_embeds = self.query_attention(
                        query_embeds.unsqueeze(0), key=map_query.unsqueeze(0)
                    )
                    query_embeds_list.append(query_embeds.squeeze(0))
            object_query_embeds = torch.stack(query_embeds_list, dim=0)
        elif self.query_embed_type == "map_instance_pos":
            local_map_input = (
                local_map.permute(1, 0, 2)
                .contiguous()
                .view(bs, self.bev_h, self.bev_w, -1)
                .permute(0, 3, 1, 2)
            )
            local_map_input = F.interpolate(
                local_map_input,
                [
                    self.bev_h // self.map_query_scale,
                    self.bev_w // self.map_query_scale,
                ],
            ).permute(0, 2, 3, 1)
            local_map_pos = F.interpolate(
                bev_pos,
                [
                    self.bev_h // self.map_query_scale,
                    self.bev_w // self.map_query_scale,
                ],
            ).permute(0, 2, 3, 1)
            local_map_valid, _ = torch.max(local_map_input, dim=-1)
            query_embeds_list = []
            for i in range(bs):
                valid_index = local_map_valid[i] > 0.8
                if torch.equal(
                    valid_index,
                    torch.zeros(
                        valid_index.shape, dtype=torch.bool, device=valid_index.device
                    ),
                ):
                    query_embeds_list.append(
                        self.instance_embedding.weight[0:num_vec].to(dtype)
                    )
                else:
                    map_pos_embed = local_map_pos[i][valid_index]
                    map_label = local_map_input[i][valid_index]
                    map_label_embed = self.label_encoder(map_label)
                    map_query = torch.cat([map_pos_embed, map_label_embed], dim=-1)
                    query_embeds = self.instance_embedding.weight[0:num_vec].to(dtype)
                    query_embeds = self.query_attention(
                        query_embeds.unsqueeze(0), key=map_query.unsqueeze(0)
                    )
                    query_embeds_list.append(query_embeds.squeeze(0))
            object_query_embeds = torch.stack(query_embeds_list, dim=0)
        self_attn_mask = torch.zeros([num_vec, num_vec]).bool().to(mlvl_feats[0].device)
        self_attn_mask[self.num_vec_one2one :, 0 : self.num_vec_one2one] = True
        self_attn_mask[0 : self.num_vec_one2one, self.num_vec_one2one :] = True
        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                local_map=local_map.permute(1, 2, 0),
            )["bev"]
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                local_map=local_map,
                uncertainty_map=uncertainty_pred,
                num_vec=num_vec,
                num_pts_per_vec=self.num_pts_per_vec,
                self_attn_mask=self_attn_mask,
                reshape=False,
            )
        bev_embed, depth, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                if "pts" not in self.query_embed_type:
                    reference = init_reference
                else:
                    reference = (
                        init_reference[..., 0:2]
                        if not self.z_cfg["gt_z_flag"]
                        else init_reference[..., 0:3]
                    )
            elif "pts" not in self.query_embed_type:
                reference = inter_references[lvl - 1]
            else:
                reference = (
                    inter_references[lvl - 1][..., 0:2]
                    if not self.z_cfg["gt_z_flag"]
                    else inter_references[lvl - 1][..., 0:3]
                )
            reference = inverse_sigmoid(reference)
            if "pts" not in self.query_embed_type:
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])
                tmp[..., :] += reference[..., :]
                tmp = tmp.sigmoid()
            else:
                outputs_class = self.cls_branches[lvl](
                    hs[lvl].view(bs, num_vec, self.num_pts_per_vec, -1).mean(2)
                )
                tmp = self.reg_branches[lvl](hs[lvl])
                tmp = tmp[..., 0:2] if not self.z_cfg["gt_z_flag"] else tmp[..., 0:3]
                tmp += reference
                tmp = tmp.sigmoid()
            outputs_coord, outputs_pts_coord = self.transform_box(tmp, num_vec=num_vec)
            outputs_classes_one2one.append(outputs_class[:, 0 : self.num_vec_one2one])
            outputs_coords_one2one.append(outputs_coord[:, 0 : self.num_vec_one2one])
            outputs_pts_coords_one2one.append(
                outputs_pts_coord[:, 0 : self.num_vec_one2one]
            )
            outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one :])
            outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one :])
            outputs_pts_coords_one2many.append(
                outputs_pts_coord[:, self.num_vec_one2one :]
            )
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)
        outputs_seg = None
        outputs_pv_seg = None
        if self.aux_seg["use_aux_seg"]:
            seg_bev_embed = (
                bev_embed.permute(1, 0, 2)
                .view(bs, self.bev_h, self.bev_w, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if self.aux_seg["bev_seg"]:
                outputs_seg = self.seg_head(seg_bev_embed)
            bs, num_cam, embed_dims, feat_h, feat_w = mlvl_feats[-1].shape
            if self.aux_seg["pv_seg"]:
                outputs_pv_seg = self.pv_seg_head(mlvl_feats[-1].flatten(0, 1))
                outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes_one2one,
            "all_bbox_preds": outputs_coords_one2one,
            "all_pts_preds": outputs_pts_coords_one2one,
            "hs": hs,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "depth": depth,
            "seg": outputs_seg,
            "pv_seg": outputs_pv_seg,
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_cls_scores=None,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                seg=None,
                pv_seg=None,
            ),
        }
        if self.enable_error_detection and "uncertainty_pred" in locals():
            outs["uncertainty_pred"] = uncertainty_pred
        return outs

    def transform_box(self, pts, num_vec=50, y_first=False):
        if self.z_cfg["gt_z_flag"]:
            pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec, 3)
        else:
            pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == "minmax":
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        return (bbox, pts_reshape)

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        pts_pred,
        gt_labels,
        gt_bboxes,
        gt_shifts_pts,
        gt_bboxes_ignore=None,
    ):
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.assigner.assign(
            bbox_pred,
            cls_score,
            pts_pred,
            gt_bboxes,
            gt_labels,
            gt_shifts_pts,
            gt_bboxes_ignore,
        )
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[
                sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds
            ]
        pts_targets = pts_pred.new_zeros(
            (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2))
        )
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[
            sampling_result.pos_assigned_gt_inds, assigned_shift, :, :
        ]
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pts_targets,
            pts_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        pts_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        pts_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )
        normalized_pts_targets = (
            normalize_2d_pts(pts_targets, self.pc_range)
            if not self.z_cfg["gt_z_flag"]
            else normalize_3d_pts(pts_targets, self.pc_range)
        )
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(
                pts_preds,
                size=self.num_pts_per_gt_vec,
                mode="linear",
                align_corners=True,
            )
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )
        dir_weights = pts_weights[:, : -self.dir_interval, 0]
        denormed_pts_preds = (
            denormalize_2d_pts(pts_preds, self.pc_range)
            if not self.z_cfg["gt_z_flag"]
            else denormalize_3d_pts(pts_preds, self.pc_range)
        )
        denormed_pts_preds_dir = (
            denormed_pts_preds[:, self.dir_interval :, :]
            - denormed_pts_preds[:, : -self.dir_interval, :]
        )
        pts_targets_dir = (
            pts_targets[:, self.dir_interval :, :]
            - pts_targets[:, : -self.dir_interval, :]
        )
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )
        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return (loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir)

    @force_fp32(apply_to="preds_dicts")
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_seg_mask,
        gt_pv_seg_mask,
        preds_dicts,
        gt_bboxes_ignore=None,
        img_metas=None,
        local_map_mask=None,
    ):
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        all_pts_preds = preds_dicts["all_pts_preds"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]
        enc_pts_preds = preds_dicts["enc_pts_preds"]
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list
        ]
        if self.gt_shift_pts_pattern == "v0":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v1":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v2":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v3":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v4":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_pts_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )
        loss_dict = dict()
        if (
            self.enable_error_detection
            and "uncertainty_pred" in preds_dicts
            and (local_map_mask is not None)
        ):
            uncertainty_pred = preds_dicts["uncertainty_pred"]
            gt_error_mask = local_map_mask.clone()
            gt_labels = torch.zeros_like(gt_error_mask, dtype=torch.long)
            gt_labels[gt_error_mask == 0] = 0
            gt_labels[gt_error_mask == 1] = 1
            gt_labels[gt_error_mask >= 2] = 2
            valid_mask = torch.ones_like(gt_error_mask, dtype=torch.bool)
            if valid_mask.sum() > 0:
                pred_reshaped = uncertainty_pred.permute(0, 2, 3, 1).reshape(-1, 3)
                gt_reshaped = gt_labels.view(-1)
                if self.uncertainty_loss_type == "cb":
                    unique_classes, counts = torch.unique(
                        gt_reshaped, return_counts=True
                    )
                    samples_per_class = torch.zeros(3, device=gt_reshaped.device)
                    for cls, count in zip(unique_classes, counts):
                        if cls < 3:
                            samples_per_class[cls] = count
                    total_samples = samples_per_class.sum()
                    if total_samples > 0:
                        samples_per_class = samples_per_class / total_samples
                        samples_per_class = samples_per_class.clamp(min=1e-06)
                    else:
                        samples_per_class = torch.tensor(
                            [0.33, 0.33, 0.34], device=gt_reshaped.device
                        )
                    effective_num = 1.0 - torch.pow(self.cb_beta, samples_per_class)
                    weights = (1.0 - self.cb_beta) / effective_num
                    weights = weights / weights.sum() * len(weights)
                    self.loss_error_detection.weights = weights
                loss_uncertainty = self.loss_error_detection(pred_reshaped, gt_reshaped)
            else:
                loss_uncertainty = torch.tensor(
                    0.0,
                    device=uncertainty_pred.device,
                    dtype=torch.float64,
                    requires_grad=True,
                )
            loss_dict["loss_uncertainty"] = loss_uncertainty
            with torch.no_grad():
                uncertainty_pred_prob = F.softmax(uncertainty_pred, dim=1)
                predicted_labels = torch.argmax(uncertainty_pred_prob, dim=1)
                accuracy = (predicted_labels == gt_labels).float().mean()
                loss_dict["uncertainty_accuracy"] = accuracy
                for cls_idx in range(3):
                    cls_name = ["correct", "error", "no_map"][cls_idx]
                    cls_mask = gt_labels == cls_idx
                    if cls_mask.sum() > 0:
                        cls_accuracy = (
                            (predicted_labels[cls_mask] == gt_labels[cls_mask])
                            .float()
                            .mean()
                        )
                        loss_dict[f"uncertainty_accuracy_{cls_name}"] = cls_accuracy
                    else:
                        loss_dict[f"uncertainty_accuracy_{cls_name}"] = torch.tensor(
                            0.0, device=uncertainty_pred.device, dtype=torch.float64
                        )
                confidence_map = uncertainty_pred_prob[:, 0, :, :]
                for cls_idx in range(3):
                    cls_name = ["correct", "error", "no_map"][cls_idx]
                    tp = (
                        ((predicted_labels == cls_idx) & (gt_labels == cls_idx))
                        .float()
                        .sum()
                    )
                    fp = (
                        ((predicted_labels == cls_idx) & (gt_labels != cls_idx))
                        .float()
                        .sum()
                    )
                    fn = (
                        ((predicted_labels != cls_idx) & (gt_labels == cls_idx))
                        .float()
                        .sum()
                    )
                    precision = tp / (tp + fp + 1e-08)
                    recall = tp / (tp + fn + 1e-08)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-08)
                    loss_dict[f"uncertainty_precision_{cls_name}"] = precision
                    loss_dict[f"uncertainty_recall_{cls_name}"] = recall
                    loss_dict[f"uncertainty_f1_{cls_name}"] = f1
                gt_confidence = (gt_labels == 0).float()
        if self.aux_seg["use_aux_seg"]:
            if self.aux_seg["bev_seg"]:
                if preds_dicts["seg"] is not None:
                    seg_output = preds_dicts["seg"]
                    num_imgs = seg_output.size(0)
                    seg_gt = torch.stack(
                        [gt_seg_mask[i] for i in range(num_imgs)], dim=0
                    )
                    loss_seg = self.loss_seg(seg_output, seg_gt.float())
                    loss_dict["loss_seg"] = loss_seg
            if self.aux_seg["pv_seg"]:
                if preds_dicts["pv_seg"] is not None:
                    pv_seg_output = preds_dicts["pv_seg"]
                    num_imgs = pv_seg_output.size(0)
                    pv_seg_gt = torch.stack(
                        [gt_pv_seg_mask[i] for i in range(num_imgs)], dim=0
                    )
                    loss_pv_seg = self.loss_pv_seg(pv_seg_output, pv_seg_gt.float())
                    loss_dict["loss_pv_seg"] = loss_pv_seg
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            (
                enc_loss_cls,
                enc_losses_bbox,
                enc_losses_iou,
                enc_losses_pts,
                enc_losses_dir,
            ) = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                enc_pts_preds,
                gt_bboxes_list,
                binary_labels_list,
                gt_pts_list,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_losses_iou"] = enc_losses_iou
            loss_dict["enc_losses_pts"] = enc_losses_pts
            loss_dict["enc_losses_dir"] = enc_losses_dir
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_pts"] = losses_pts[-1]
        loss_dict["loss_dir"] = losses_dir[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(
            losses_cls[:-1],
            losses_bbox[:-1],
            losses_iou[:-1],
            losses_pts[:-1],
            losses_dir[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_pts"] = loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_dir"] = loss_dir_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to="preds_dicts")
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]
            ret_list.append([bboxes, scores, labels, pts])
        return ret_list

    def _extract_uncertainty_confidence_for_points(self, vector_maps, uncertainty_pred):
        bs, m, num_pts, _ = vector_maps.shape
        _, _, bev_h, bev_w = uncertainty_pred.shape
        points_normalized = vector_maps[..., :2]
        points_grid = points_normalized.clone()
        points_grid[..., 0] = points_grid[..., 0] * (bev_w - 1)
        points_grid[..., 1] = points_grid[..., 1] * (bev_h - 1)
        points_grid = torch.clamp(points_grid, 0, min(bev_h - 1, bev_w - 1))
        points_grid_int = points_grid.long()
        uncertainty_prob = F.softmax(uncertainty_pred, dim=1)
        prob_correct = uncertainty_prob[:, 0, :, :]
        prob_error = uncertainty_prob[:, 1, :, :]
        prob_no_map = uncertainty_prob[:, 2, :, :]
        confidence_map = prob_correct - prob_error
        not_have_map_mask = prob_no_map > 0.5
        confidence_map[not_have_map_mask] = 0.0
        confidence_scores = torch.zeros(
            (bs, m, num_pts), device=vector_maps.device, dtype=torch.float32
        )
        for b in range(bs):
            for i in range(m):
                for p in range(num_pts):
                    x_idx = points_grid_int[b, i, p, 0]
                    y_idx = points_grid_int[b, i, p, 1]
                    confidence_scores[b, i, p] = confidence_map[b, y_idx, x_idx]
        return confidence_scores
