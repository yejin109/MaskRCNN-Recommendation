import os
import json

with open('data/DF2/validation/validation.json') as sf:
    sample = json.load(sf)


categories = list()
categories.append({'id': 1, 'name': '상의'})
categories.append({'id': 2, 'name': '치마'})
categories.append({'id': 3, 'name': '바지'})
categories.append({'id': 4, 'name': "아우터"})

global_category_to_idx = dict()

for category in categories:
    global_category_to_idx[category['name']] = category['id']


file_names = os.listdir('data/musinsa_codimap/annos/')
files = dict()
images = []
annotations = []

img_count = 1
anno_count = 1

for file_idx, file_name in enumerate(file_names):
    with open(f'data/musinsa_codimap/annos/{file_names[file_idx]}', encoding='UTF-8') as f:
        chunk = json.load(f)
        local_idx_to_category = dict()

        for category in chunk['categories']:
            local_idx_to_category[category['id']] = category['name']

        for image in chunk['images']:
            past_id = image['id']
            for anno in chunk['annotations']:
                if anno['image_id'] != past_id:
                    continue
                anno['id'] = anno_count
                anno['image_id'] = img_count

                anno_category = local_idx_to_category[anno['category_id']]

                anno['category_id'] = global_category_to_idx[anno_category]

                annotations.append(anno)
                anno_count += 1

            image['id'] = img_count
            images.append(image)
            img_count += 1
    files[file_name] = chunk


dataset_json = dict()
dataset_json['info'] = dict()
dataset_json['images'] = images
dataset_json['annotations'] = annotations
dataset_json['categories'] = categories

json_name = f'data/musinsa_codimap/musinsa_dataset.json'
with open(json_name, 'w') as f:
    json.dump(dataset_json, f)

print()
