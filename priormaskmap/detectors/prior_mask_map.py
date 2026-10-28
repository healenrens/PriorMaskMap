import copy
import torch
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from priormaskmap.components.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from priormaskmap.priors.prior_map import GlobalMap


@DETECTORS.register_module()
class PriorMaskMap(MVXTwoStageDetector):
    """Predict map polylines from camera images and offline priors."""

    def __init__(
        self,
        img_backbone,
        img_neck,
        pts_bbox_head,
        global_map_cfg,
        use_grid_mask=False,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):
        super().__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.global_map = GlobalMap(global_map_cfg)
        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to="img", out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        lidar_feat,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        local_map=None,
        gt_depth=None,
        gt_seg_mask=None,
        gt_pv_seg_mask=None,
        local_map_mask=None,
        vector_maps_gt=None,
        vector_maps_modified=None,
    ):
        outs = self.pts_bbox_head(
            pts_feats,
            lidar_feat,
            img_metas,
            local_map,
            False,
            vector_maps_gt,
            vector_maps_modified,
        )
        depth = outs.pop("depth")
        bev_embed = outs.get("bev_embed")
        batchsize = bev_embed.shape[0]
        losses = dict()
        if gt_depth is not None and depth is not None:
            loss_depth = self.pts_bbox_head.transformer.encoder.get_depth_loss(
                gt_depth, depth
            )
            if digit_version(TORCH_VERSION) >= digit_version("1.8"):
                loss_depth = torch.nan_to_num(loss_depth)
            losses.update(loss_depth=loss_depth)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_seg_mask, gt_pv_seg_mask, outs]
        losses_pts = self.pts_bbox_head.loss(
            *loss_inputs, img_metas=img_metas, local_map_mask=local_map_mask
        )
        losses.update(losses_pts)
        k_one2many = self.pts_bbox_head.k_one2many
        if k_one2many > 0:
            multi_gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
            multi_gt_labels_3d = copy.deepcopy(gt_labels_3d)
            for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(
                zip(multi_gt_bboxes_3d, multi_gt_labels_3d)
            ):
                each_gt_bboxes_3d.instance_list = (
                    each_gt_bboxes_3d.instance_list * k_one2many
                )
                each_gt_bboxes_3d.instance_labels = (
                    each_gt_bboxes_3d.instance_labels * k_one2many
                )
                multi_gt_labels_3d[i] = each_gt_labels_3d.repeat(k_one2many)
            one2many_outs = outs["one2many_outs"]
            loss_one2many_inputs = [
                multi_gt_bboxes_3d,
                multi_gt_labels_3d,
                gt_seg_mask,
                gt_pv_seg_mask,
                one2many_outs,
            ]
            loss_dict_one2many = self.pts_bbox_head.loss(
                *loss_one2many_inputs, img_metas=img_metas
            )
            lambda_one2many = self.pts_bbox_head.lambda_one2many
            for key, value in loss_dict_one2many.items():
                if key + "_one2many" in losses.keys():
                    losses[key + "_one2many"] += value * lambda_one2many
                else:
                    losses[key + "_one2many"] = value * lambda_one2many
        return losses

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @force_fp32(apply_to=("img", "points", "local_map"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        gt_depth=None,
        gt_seg_mask=None,
        gt_pv_seg_mask=None,
    ):
        lidar_feat = None
        len_queue = img.size(1)
        img = img[:, -1, ...]
        batch_data = {
            "sample_idx": [meta[0]["sample_idx"] for meta in img_metas],
            "city_name": [meta[0]["map_location"] for meta in img_metas],
            "lidar2global": [meta[0]["lidar2global"] for meta in img_metas],
        }
        img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        (
            local_map,
            _,
            local_map_mask,
            original_maps,
            modified_maps,
            map_available,
            vector_maps_gt,
            vector_maps_modified,
        ) = self.global_map.get_patch_mix_data(batch_data, img_feats[0].device, "train")
        losses_pts = self.forward_pts_train(
            img_feats,
            lidar_feat,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            local_map,
            gt_depth,
            gt_seg_mask,
            gt_pv_seg_mask,
            local_map_mask,
            vector_maps_gt,
            vector_maps_modified,
        )
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        img = [img] if img is None else img
        points = [points] if points is None else points
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], None, **kwargs
        )
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to("cpu"),
        )
        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()
        return result_dict

    def simple_test_pts(
        self,
        x,
        lidar_feat,
        img_metas,
        local_map=None,
        vector_maps_modified=None,
        rescale=False,
    ):
        outs = self.pts_bbox_head(
            x,
            lidar_feat,
            img_metas,
            local_map=local_map,
            vector_maps_modified=vector_maps_modified,
        )
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]
        return (outs["bev_embed"], bbox_results)

    def simple_test(
        self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs
    ):
        lidar_feat = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        batch_data = {
            "sample_idx": [img_metas[0]["sample_idx"]],
            "city_name": [img_metas[0]["map_location"]],
            "lidar2global": [img_metas[0]["lidar2global"]],
        }
        (
            local_map,
            _,
            local_map_mask,
            original_maps,
            modified_maps,
            map_available,
            vector_maps_gt,
            vector_maps_modified,
        ) = self.global_map.get_patch_mix_data(batch_data, img_feats[0].device, "val")
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats,
            lidar_feat,
            img_metas,
            modified_maps,
            vector_maps_modified,
            rescale=rescale,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return (new_prev_bev, bbox_list)
