import os
import cv2
import json
import torch
from torchvision import transforms
from tqdm import tqdm

category_name = dict()
category_name['상의'] = 'top'
category_name['치마'] = 'skirt'
category_name['바지'] = 'trouser'
category_name['아우터'] = 'outwear'


def get_recom_transform():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 이미지 지터링(밝기, 대조, 채비, 색조)
            transforms.RandomHorizontalFlip(p = 0.5), # p확률로 이미지 좌우반전
            transforms.RandomVerticalFlip(p = 0.5), # p확률로 상하반전
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 이미지 지터링(밝기, 대조, 채비, 색조)
            transforms.RandomHorizontalFlip(p = 0.5), # p확률로 이미지 좌우반전
            transforms.RandomVerticalFlip(p = 0.5), # p확률로 상하반전
        ]),
        'candidate_emb': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 이미지 지터링(밝기, 대조, 채비, 색조)
            transforms.RandomHorizontalFlip(p = 0.5), # p확률로 이미지 좌우반전
            transforms.RandomVerticalFlip(p = 0.5), # p확률로 상하반전
        ]),
    }
    return data_transforms


# TODO: 카테고리 지정
def categorize(root_path):
    with open(f"{root_path}/data/recom_train/total_dataset.json", encoding='utf-8') as f:
        info = json.load(f)
    categories = info['categories']
    global_idx_to_category = dict()

    for category in categories:
        global_idx_to_category[category['id']] = category['name']

    past_images = []
    for anno in info['annotations']:
        image_id = anno['image_id']
        if image_id in past_images:
            continue
        past_images.append(image_id)

        category_id = anno['category_id']
        category = category_name[global_idx_to_category[category_id]]
        file_name = info['images'][image_id-1]['file_name']

        image = cv2.imread(f"{root_path}/data/recom_train/image/{file_name}", cv2.COLOR_BGR2RGB)
        dir_path = f"{root_path}/data/recom_train/{category}"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        cv2.imwrite(f'{dir_path}/{file_name}', image)


def aggregate_emb(root_path):
    total = torch.Tensor()
    for item in tqdm(os.listdir(f"{root_path}/save/recom_item_output/candidate_emb")):
        emb = torch.load(f"{root_path}/save/recom_item_output/candidate_emb/{item}")
        total = torch.cat((total, emb), dim=0)
    torch.save(total, f"{root_path}/save/recom_item_output/total.pt")
    return total
