# # TODO: TF로 구현한 코드 여기에 정리하기
#
# import os
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
#
# from dataset import ODDataset
# from config import MaskConfig
# from util.utils import tensor2img, apply_mask, Compose, Resize, ToTensor, collate_fn
# from util.deepfashion2_to_coco import toCOCO
# from model import EmbeddingExtractor, MaskIndexer
#
# import torch
# from torch.optim import SGD
# from torch.utils.data import DataLoader
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#
# root_path = os.getcwd()
# config = MaskConfig(root_path)
#
# # COCO 형태로 바꾸기
# # toCOCO('validation', root_path)
#
# # Hyper-parameter
# lr = config.lr
# weight_decay = config.weight_decay
# num_epochs = config.num_epochs
# batch_size = config.batch_size
# hidden_layer = config.hidden_layer
#
# classes = config.classes
# num_classes = config.NUM_CLASSES
# max_size = config.max_size
# score_threshold = config.score_threshold
# device = config.device
#
# # Model
# model = maskrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
#
# model.to(device)
#
# # Dataset
# transform = Compose([Resize((max_size, max_size)), ToTensor()])
# train_dataset = ODDataset(config.musinsa_json_dir, config.musinsa_img_dir, device, transforms=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
# optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#
# # Train
# model.train()
# print(f"한 에폭당 iteration 수 : {len(train_loader)}")
# loss_per_iter = []
#
# for epoch in range(num_epochs):
#     for i, (images, targets) in tqdm(enumerate(train_loader)):
#         optimizer.zero_grad()
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         losses = model(images, targets)
#         loss = sum(loss for loss in losses.values())
#         loss_per_iter.append(loss.detach().cpu().numpy())
#
#         print(
#             f"{epoch}, {i}, C: {losses['loss_classifier'].item():.5f}, M: {losses['loss_mask'].item():.5f}, "
#             f"B: {losses['loss_box_reg'].item():.5f}, O: {losses['loss_objectness'].item():.5f}, T: {loss.item():.5f}")
#
#         loss.backward()
#         optimizer.step()
#     print()
#
# torch.save(model.state_dict(), 'save/model.pt')
#
# plt.figure()
# plt.plot(loss_per_iter)
# plt.show()
#
#
# # Test
#
# model.load_state_dict(torch.load('save/model.pt'))
#
# box_predictor_param = model.roi_heads.box_predictor.state_dict()
# embedding_extractor = EmbeddingExtractor(in_features, num_classes, box_predictor_param)
# model.roi_heads.box_predictor = embedding_extractor
#
# # mask_predictor_param = model.roi_heads.mask_predictor.state_dict()
# # mask_indexer = MaskIndexer(in_features_mask, hidden_layer, num_classes, mask_predictor_param)
# # model.roi_heads.mask_predictor = mask_indexer
#
# test_dataset = ODDataset(config.musinsa_json_dir, config.musinsa_img_dir, device, transforms=transform)
# test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
#
# embedding_label = []
#
# model.eval()
# with torch.no_grad():
#     for i, (images, targets) in enumerate(test_loader):
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         result = model(images)
#
#         image = tensor2img(images[0])
#         scores = list(result[0]['scores'].detach().cpu().numpy())
#
#         if len(scores) != 0:
#             # TODO: 원래는 이렇게 해야함
#             # thresholded_preds_inidices = [scores.index(i) for i in scores if i > score_threshold]
#             thresholded_preds_inidices = [np.argmax(scores)]
#
#             thresholded_preds_count = len(thresholded_preds_inidices)
#
#             mask = result[0]['masks']
#             mask = mask[:thresholded_preds_count]
#
#             labels = result[0]['labels']
#
#             boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes']]
#             boxes = boxes[:thresholded_preds_count]
#
#             mask = mask.data.float().cpu().numpy()
#             final_labels = apply_mask(image, mask, labels, boxes, i, classes)
#
#         else:
#             final_labels = [-1]
#         embedding_label.append(final_labels)
#
# embedding_label = np.array(embedding_label)
# np.savetxt('save/labels.txt', embedding_label)
#
