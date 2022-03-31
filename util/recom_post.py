import os
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def name_filling(source, style_id):
    if source == 'codibook':
        return f"codibook_style_{style_id}"
    else:
        return style_id


def item_sorting(root_path: str):
    sources = ['musinsa_codimap', 'codibook']
    item_search = dict()
    for src in sources:
        item_directory = f"{root_path}/data/recom_test/{src}/info/info_item/"
        style_directory = f"{root_path}/data/recom_test/{src}/info/info_style/"

        path_info_items = os.listdir(item_directory)
        for path_info_item in tqdm(path_info_items):
            with open(item_directory + path_info_item, encoding='utf-8') as f:
                info_item = json.load(f)
            if 'style_id' not in info_item.keys():
                continue

            if info_item['title'] not in item_search.keys():
                with open(style_directory+name_filling(src, info_item['style_id'])+'.json', encoding='utf-8') as style_f:
                    style_info = json.load(style_f)
                item_search[info_item['title']] = {'styles': [info_item['style_id']], 'item_link': info_item['link'],
                                                   'style_link': style_info['link']}

            else:
                item_search[info_item['title']]['styles'].append(info_item['style_id'])

    with open(f"{root_path}/data/recom_test/item_to_outfit.json", 'w', encoding='utf-8') as f:
        json.dump(item_search, f, indent="\t", ensure_ascii=False)


# TODO: 코디북과 무신사 비율 조정


# TODO: 추천된 아이템 --> 코디 변환 코딩
def item_to_outfit(item):
    outfit = item
    return outfit


def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(inputs.cpu().data[j].T)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)