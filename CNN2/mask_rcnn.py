from dataset import ImgDataset
from utils import tensor2img, apply_mask

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

batch_size = 1
score_threshold = 0.965

transform = [transforms.ToTensor()]
dataset = ImgDataset(transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = maskrcnn_resnet50_fpn(pretrained=True).to(device)

model.eval()
for idx, data in enumerate(dataloader):

    result = model(data)

    image = tensor2img(data)
    scores = list(result[0]['scores'].detach().cpu().numpy())
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > score_threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    mask = result[0]['masks']
    mask = mask[:thresholded_preds_count]
    labels = result[0]['labels']
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes']]
    boxes = boxes[:thresholded_preds_count]

    mask = mask.data.float().cpu().numpy()

    apply_mask(image, mask, labels, boxes, idx)
print()
