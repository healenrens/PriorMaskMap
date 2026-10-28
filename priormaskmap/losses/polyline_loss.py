# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models import weighted_loss
import mmcv
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import functools


@mmcv.jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if not avg_factor is None:
        if reduction == "mean":
            loss = loss.sum()
            loss = loss / avg_factor
    return loss


def custom_weighted_dir_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    loss = torch.abs(pred - target)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction="none")
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0, 1), target.flatten(0, 1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss


@LOSSES.register_module()
class PriorPtsDirCosLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(PriorPtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        reduction = reduction_override if reduction_override else self.reduction
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_dir


@LOSSES.register_module()
class PriorPtsL1Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(PriorPtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_bbox


@MATCH_COST.register_module()
class PriorOrderedPtsL1Cost(object):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
