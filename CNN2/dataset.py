import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ImgDataset(Dataset):
    def __init__(self, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        files = []
        file_list = os.listdir('data/')
        for file in file_list:
            # 이 때 Img의 길이를 일치시켜야 하는 문제 존재
            img = Image.open(f'data/{file}').resize((1024, 1024), Image.LANCZOS)
            files.append(self.transform(img))

        self.files = files

    def __getitem__(self, index):
        return self.files[index].to(device)

    def __len__(self):
        return len(self.files)
