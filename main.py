import os
from config import RecomConfig

import main_mask
from dataset.mask_dataset import ODDataset

import main_recom
from dataset.recom_dataset import get_recom_data_setting
from util.recom_post import visualize_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.getcwd()

########################################################################################################################
# Mask
########################################################################################################################

mask_dataset = ODDataset()

########################################################################################################################
# Recommendation
########################################################################################################################

# Setup
recom_config = RecomConfig(root_path)

recom_data_dir = recom_config.train_data_dir

image_datasets, dataloaders, dataset_sizes, class_names = get_recom_data_setting(recom_data_dir)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train
main_recom.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, dataloaders)

# Test
visualize_model(model_ft, dataloaders, class_names)
