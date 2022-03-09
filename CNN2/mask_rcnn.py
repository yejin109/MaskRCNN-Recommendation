from dataset import ImgDataset, FashionDataset
from util.utils import tensor2img, apply_mask, Compose, Resize, ToTensor, collate_fn
import config

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# HyperParameter
lr = config.lr
num_epochs = config.num_epochs
batch_size = config.batch_size
hidden_layer = config.hidden_layer

classes = config.classes
num_classes = len(classes)
max_size = config.max_size
score_threshold = config.score_threshold

# Model
model = maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, len(classes)+1)


transform = Compose([Resize((max_size, max_size)), ToTensor()])
train_dataset = FashionDataset('data/fashion/train.json', transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = Adam(params, lr=lr, weight_decay=1e-5)

# Train
model.train()
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())

        print(
            f"{epoch}, {i}, C: {losses['loss_classifier'].item():.5f}, M: {losses['loss_mask'].item():.5f}, "
            f"B: {losses['loss_box_reg'].item():.5f}, O: {losses['loss_objectness'].item():.5f}, T: {loss.item():.5f}")
        loss.backward()
        optimizer.step()

# Test
test_dataset = ImgDataset(transform, max_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
model.eval()
for idx, data in enumerate(test_dataloader):

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
