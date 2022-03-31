import os
from torchvision import datasets
from util.recom_pre import get_recom_transform
from torch.utils.data import DataLoader


# 찬영님 기준
def get_recom_data_setting(data_dir: str):
    data_transforms = get_recom_transform()
    image_datasets = {x: datasets.ImageFolder(f"{data_dir}",
                                              data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return image_datasets, dataloaders, dataset_sizes, class_names


# 재현님 기준

