import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.mask_dataset import ODDataset
from util.mask_pre import collate_fn
from util.mask_post import tensor2img, apply_mask
from model.mask_model import EmbeddingExtractor

import torch
from torch.utils.data import DataLoader


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Train
def train_mask_model(model, train_loader, val_loader, num_epochs, optimizer, root_path):
    print(f"한 에폭당 iteration 수 : {len(train_loader)}")
    losses_summary = dict()
    losses_summary['loss_classifier'] = []
    loss_classifier = []
    losses_summary['loss_mask'] = []
    loss_mask = []
    losses_summary['loss_box_reg'] = []
    loss_box_reg = []
    losses_summary['loss_objectness'] = []
    loss_objectness = []
    losses_summary['total'] = []
    total = []

    val_losses_summary = dict()
    val_losses_summary['loss_classifier'] = []
    val_loss_classifier = []
    val_losses_summary['loss_mask'] = []
    val_loss_mask = []
    val_losses_summary['loss_box_reg'] = []
    val_loss_box_reg = []
    val_losses_summary['loss_objectness'] = []
    val_loss_objectness = []
    val_losses_summary['total'] = []
    val_total = []

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            loss_classifier.append(losses['loss_classifier'].item())
            loss_mask.append(losses['loss_mask'].item())
            loss_box_reg.append(losses['loss_box_reg'].item())
            loss_objectness.append(losses['loss_objectness'].item())
            total.append(loss.item())

            # print(
            #     f"{epoch}, {i}, C: {losses['loss_classifier'].item():.5f}, M: {losses['loss_mask'].item():.5f}, "
            #     f"B: {losses['loss_box_reg'].item():.5f}, O: {losses['loss_objectness'].item():.5f}, T: {loss.item():.5f}")

            loss.backward()
            optimizer.step()
        losses_summary['loss_classifier'].append(np.mean(loss_classifier))
        losses_summary['loss_mask'].append(np.mean(loss_mask))
        losses_summary['loss_box_reg'].append(np.mean(loss_box_reg))
        losses_summary['loss_objectness'].append(np.mean(loss_objectness))
        losses_summary['total'].append(np.mean(total))

        with torch.no_grad():
            for i, (images, targets) in tqdm(enumerate(val_loader)):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                losses = model(images, targets)
                loss = sum(loss for loss in losses.values())
                val_loss_classifier.append(losses['loss_classifier'].item())
                val_loss_mask.append(losses['loss_mask'].item())
                val_loss_box_reg.append(losses['loss_box_reg'].item())
                val_loss_objectness.append(losses['loss_objectness'].item())
                val_total.append(loss.item())
        val_losses_summary['loss_classifier'].append(np.mean(val_loss_classifier))
        val_losses_summary['loss_mask'].append(np.mean(val_loss_mask))
        val_losses_summary['loss_box_reg'].append(np.mean(val_loss_box_reg))
        val_losses_summary['loss_objectness'].append(np.mean(val_loss_objectness))
        val_losses_summary['total'].append(np.mean(val_total))

    losses_summary = pd.DataFrame.from_dict(losses_summary)
    losses_summary.to_csv(f'{root_path}/save/log/mask_train_log.csv')

    val_losses_summary = pd.DataFrame.from_dict(val_losses_summary)
    val_losses_summary.to_csv(f'{root_path}/save/log/mask_val_log.csv')


    # torch.save(model.state_dict(), 'save/mask_model/model_mask.pt')


# Test
def test_mask_model(model, num_classes, json_path, image_dir_path, transform, classes):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    box_predictor_param = model.roi_heads.box_predictor.state_dict()
    embedding_extractor = EmbeddingExtractor(in_features, num_classes, box_predictor_param)
    model.roi_heads.box_predictor = embedding_extractor
    model.load_state_dict(torch.load('save/mask_model/model_mask.pt'))

    # mask_predictor_param = model.roi_heads.mask_predictor.state_dict()
    # mask_indexer = MaskIndexer(in_features_mask, hidden_layer, num_classes, mask_predictor_param)
    # model.roi_heads.mask_predictor = mask_indexer

    test_dataset = ODDataset(json_path, image_dir_path, device, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(test_loader)):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            result = model(images, targets)

            image = tensor2img(images[0])
            scores = list(result[0]['scores'].detach().cpu().numpy())

            # TODO: 원래는 이렇게 해야함
            # thresholded_preds_inidices = [scores.index(i) for i in scores if i > score_threshold]
            thresholded_preds_inidices = [np.argmax(scores)]

            thresholded_preds_count = len(thresholded_preds_inidices)

            mask = result[0]['masks']
            mask = mask[:thresholded_preds_count]

            labels = result[0]['labels']

            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes']]
            boxes = boxes[:thresholded_preds_count]

            mask = mask.data.float().cpu().numpy()
            apply_mask(image, mask, labels, boxes, i, classes)


