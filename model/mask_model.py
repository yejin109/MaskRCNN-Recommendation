import torch
import torch.nn as nn


class EmbeddingExtractor(nn.Module):
    """
    참고 : https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html
    """
    def __init__(self, in_channels, num_classes, initial):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.cls_score.weight = torch.nn.Parameter(initial['cls_score.weight'])
        self.cls_score.bias = torch.nn.Parameter(initial['cls_score.bias'])

        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.bbox_pred.weight = torch.nn.Parameter(initial['bbox_pred.weight'])
        self.bbox_pred.bias = torch.nn.Parameter(initial['bbox_pred.bias'])

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]

        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class MaskIndexer(nn.Module):
    """
    참고 : https://pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html
    """
    def __init__(self, in_channels, dim_reduced, num_classes, initial):
        super().__init__()
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.conv5_mask.weight = torch.nn.Parameter(initial['conv5_mask.weight'])
        self.conv5_mask.bias = torch.nn.Parameter(initial['conv5_mask.bias'])

        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        self.mask_fcn_logits.weight = torch.nn.Parameter(initial['mask_fcn_logits.weight'])
        self.mask_fcn_logits.bias = torch.nn.Parameter(initial['mask_fcn_logits.bias'])

        self.relu = nn.ReLU(inplace=True)

        self.idx = 0

    def forward(self, x):
        output = self.conv5_mask(x)
        output = self.relu(output)
        output = self.mask_fcn_logits(output)

        return output
