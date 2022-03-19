import os
import json

# with open('data/DF2/validation/validation.json') as sf:
#     sample = json.load(sf)

data_sources = ['codibook', 'musinsa_codimap']


categories = list()
categories.append({'id': 1, 'name': '상의'})
categories.append({'id': 2, 'name': '치마'})
categories.append({'id': 3, 'name': '바지'})
categories.append({'id': 4, 'name': "아우터"})

global_category_to_idx = dict()

for category in categories:
    global_category_to_idx[category['name']] = category['id']

total_image = []
total_annos = []

for data_source in data_sources:
    file_names = os.listdir(f'data/crawling_data/{data_source}/annos/')
    files = dict()
    images = []
    annotations = []

    img_count = 1
    anno_count = 1

    for file_idx, file_name in enumerate(file_names):
        with open(f'data/crawling_data/{data_source}/annos/{file_names[file_idx]}', encoding='UTF-8') as f:
            chunk = json.load(f)
            local_idx_to_category = dict()

            for category in chunk['categories']:
                local_idx_to_category[category['id']] = category['name']

            for image in chunk['images']:
                past_id = image['id']

                # 한 이미지 내 여러 감지 대상의 annotation이 있는 경우
                for anno in chunk['annotations']:
                    # 다른 이미지인 경우에 섞이지 않도록 처리
                    if anno['image_id'] != past_id:
                        continue

                    if 'category_id' not in anno.keys():
                        continue
                    anno['id'] = anno_count
                    anno['image_id'] = img_count

                    anno_category = local_idx_to_category[anno['category_id']]

                    anno['category_id'] = global_category_to_idx[anno_category]

                    annotations.append(anno)
                    total_annos.append(anno)
                    anno_count += 1

                image['id'] = img_count
                images.append(image)
                total_image.append(image)
                img_count += 1
        files[file_name] = chunk

    dataset_json = dict()
    dataset_json['info'] = dict()
    dataset_json['images'] = images
    dataset_json['annotations'] = annotations
    dataset_json['categories'] = categories

    json_name = f'data/crawling_data/{data_source}/{data_source}_dataset.json'
    with open(json_name, 'w') as f:
        json.dump(dataset_json, f)


dataset_json = dict()
dataset_json['info'] = dict()
dataset_json['images'] = total_image
dataset_json['annotations'] = total_annos
dataset_json['categories'] = categories

json_name = f'data/crawling_data/total_dataset.json'
with open(json_name, 'w') as f:
    json.dump(dataset_json, f)