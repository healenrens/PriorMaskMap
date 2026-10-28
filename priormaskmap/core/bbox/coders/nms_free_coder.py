import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy


def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    new_pts[..., 2:3] = pts[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return new_pts


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


@BBOX_CODERS.register_module()
class PolylineCoder(BaseBBoxCoder):
    def __init__(
        self,
        pc_range,
        z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
        voxel_size=None,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        raster_threshold=0.4,
        bev_h=200,
        bev_w=100,
        num_classes=10,
    ):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.raster_threshold = raster_threshold
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.z_cfg = z_cfg

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, pts_preds):
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]
        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range)
        final_pts_preds = (
            denormalize_2d_pts(pts_preds, self.pc_range)
            if not self.z_cfg["gt_z_flag"]
            else denormalize_3d_pts(pts_preds, self.pc_range)
        )
        final_scores = scores
        final_preds = labels
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (final_box_preds[..., :4] >= self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <= self.post_center_range[4:]).all(1)
            if self.score_threshold:
                mask &= thresh_mask
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
                "pts": pts,
            }
        return predictions_dict

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_pts_preds = preds_dicts["all_pts_preds"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i]
                )
            )
        return predictions_list
