import os
import numpy as np

from PIL import Image
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Crawling Dataset
class ImgDataset(Dataset):
    def __init__(self, max_size, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        files = []
        file_list = os.listdir('data/')
        for file in file_list:
            # 이 때 Img의 길이를 일치시켜야 하는 문제 존재
            img = Image.open(f'data/{file}').resize((max_size, max_size), Image.LANCZOS)
            files.append(self.transform(img))

        self.files = files

    def __getitem__(self, index):
        return self.files[index].to(device)

    def __len__(self):
        return len(self.files)


# DeepFashion2 Dataset
class FashionDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.coco = COCO(path)
        self.image_ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        file_name = f'./data/fashion/train/{file_name}'
        image = Image.open(file_name).convert('RGB')

        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = [x for x in self.coco.loadAnns(annot_ids) if x['image_id'] == image_id]

        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32)
        masks = np.array([self.coco.annToMask(annot) for annot in annots], dtype=np.uint8)

        area = np.array([annot['area'] for annot in annots], dtype=np.float32)
        iscrowd = np.array([annot['iscrowd'] for annot in annots], dtype=np.uint8)

        target = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)

        return image, target