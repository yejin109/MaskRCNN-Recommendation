import os
from config import RecomConfig, MaskConfig
from tqdm import tqdm

from dataset.mask_dataset import ODDataset
from util.mask_pre import get_mask_transform, collate_fn
from model.mask_model import get_mask_model
from main_mask import train_mask_model, test_mask_model

import main_recom
from dataset.recom_dataset import get_recom_data_setting
from util.recom_post import candidate_emb, recommend, get_outfit, item_sorting
from util.recom_pre import categorize, aggregate_emb
from model.recom_model import ResNet_without_fc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.getcwd()

########################################################################################################################
# Mask
########################################################################################################################

# Setup
# mask_config = MaskConfig(root_path)
# num_classes = mask_config.NUM_CLASSES
# hidden_layer = mask_config.hidden_layer
# json_path = mask_config.musinsa_json_dir
# image_dir_path = mask_config.musinsa_img_dir
# batch_size = mask_config.batch_size
# lr = mask_config.lr
# weight_decay = mask_config.weight_decay
# num_epochs = mask_config.num_epochs
# classes = mask_config.classes
#
# # dataset
# mask_transform = get_mask_transform(mask_config.max_size)
# mask_dataset = ODDataset(json_path, image_dir_path, device, mask_transform)
# mask_data_loader = DataLoader(mask_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
# # model
# mask_model = get_mask_model(num_classes, hidden_layer)
# optimizer = optim.SGD(mask_model.parameters(), lr=lr, weight_decay=weight_decay)
#
# # Train
# train_mask_model(mask_model, mask_data_loader, num_epochs, optimizer)
#
# # Test
# test_mask_model(mask_model, num_classes, json_path, image_dir_path, mask_transform, classes)

########################################################################################################################
# Recommendation
########################################################################################################################

# Setup
recom_config = RecomConfig(root_path)
recom_num_classes = recom_config.NUM_CLASSES
recom_data_dir = recom_config.train_data_dir

# item_sorting(root_path)
# categorize(root_path)

image_datasets, dataloaders, dataset_sizes, class_names = get_recom_data_setting(recom_data_dir)
#
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, recom_num_classes)

# torch.save(model_ft.state_dict(), f"{root_path}/save/recom_model/pure.pt")
#
model_ft = model_ft.to(device)
#
criterion = nn.CrossEntropyLoss()
#
# # Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train
main_recom.train_recom_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, dataloaders)

# Embedding 생성
resnet_wo_fc = ResNet_without_fc([2, 2, 2, 2], num_ftrs, recom_num_classes, True).to(device)
resnet_wo_fc.load_state_dict(torch.load('save/recom_model/model_recom_2.pt'))

candidate_emb(resnet_wo_fc, root_path)
aggregate_emb(root_path)
#
# embeddings = torch.Tensor()
# for emb_name in tqdm(os.listdir(f'{root_path}/save/recom_item_output/candidate_emb')):
#     emb = torch.load(f'{root_path}/save/recom_item_output/candidate_emb/{emb_name}')
#     embeddings = torch.cat((embeddings, emb), dim=0)
# torch.save(embeddings, f'{root_path}/save/recom_item_output/total.pt')

# 추천
# example_path = os.listdir('save/recom_input/')[0]
#
# result = get_outfit(resnet_wo_fc, root_path, example_path)
#
print()
