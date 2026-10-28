import torch
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class PriorSimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super(PriorSimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss * self.loss_weight
