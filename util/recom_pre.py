import os
import cv2
import json
from torchvision import transforms

category_name = dict()
category_name['상의'] = 'top'
category_name['치마'] = 'skirt'
category_name['바지'] = 'trouser'
category_name['아우터'] = 'outwear'


def get_recom_transform():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'candidate_emb': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
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
