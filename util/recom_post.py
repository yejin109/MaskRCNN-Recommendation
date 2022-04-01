import os
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from util.recom_pre import get_recom_transform
from sklearn.neighbors import NearestNeighbors


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def style_name_filling(source, style_id):
    if source == 'codibook':
        return f"codibook_style_{style_id}"
    else:
        return style_id


def item_name_filling(source, style_id, item_id):
    if source == 'codibook':
        return f"codibook_item_{style_id}_{item_id}"
    else:
        return style_id


def item_sorting(root_path: str):
    sources = ['musinsa', 'codibook']
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
                with open(style_directory + style_name_filling(src, info_item['style_id']) + '.json', encoding='utf-8') as style_f:
                    style_info = json.load(style_f)
                item_search[item_name_filling(src, info_item['style_id'], info_item['item_id'])] = {'styles': [style_name_filling(src, info_item['style_id'])], 'item_link': info_item['link'],
                                                   'style_link': style_info['link']}

            else:
                item_search[item_name_filling(src, info_item['style_id'], info_item['item_id'])]['styles'].append(style_name_filling(src, info_item['style_id']))

    with open(f"{root_path}/data/recom_test/item_to_outfit.json", 'w', encoding='utf-8') as f:
        json.dump(item_search, f, indent="\t", ensure_ascii=False)


# TODO: 코디북과 무신사 비율 조정


# TODO: 추천된 아이템 --> 코디 변환 코딩
def item_to_outfit(root_path, items: list):
    styles = []
    links = []
    with open(f"{root_path}/data/recom_test/item_to_outfit.json", encoding='utf-8') as f:
        item_dict = json.load(f)
    for item in items:
        style = item_dict[item[:-4]]['styles']
        styles.append(style)
        link = item_dict[item[:-4]]['style_link']
        links.append(link)

    return styles, links


def candidate_emb(model, root_path):
    transformers = get_recom_transform()
    image_path_all = []

    model.eval()
    for fashion_images in tqdm(os.listdir(f'{root_path}/data/recom_test/image/item')):
        images_path = os.path.join(f'{root_path}/data/recom_test/image/item', fashion_images)
        image_path_all.append(images_path)
        img_RGB = Image.open(images_path).convert('RGB')

        # PIL to Tensor
        img_RGB_tensor_from_PIL = transformers['candidate_emb'](img_RGB)
        img_unsqueeze = torch.unsqueeze(img_RGB_tensor_from_PIL, 0)
        output_feature = model(img_unsqueeze.to(device))
        torch.save(output_feature.detach().cpu(), f"{root_path}/save/recom_item_output/candidate_emb/{fashion_images[:-4]}.pt")
    # imageset_total = imageset_total.to(device)
    # with torch.no_grad():
    #     output_feature = resnet_wo_fc(imageset_total)
    # print(output_feature.shape)
    # torch.Size([2,512])

    # pickle.dump(output_feature, open(f"{root_path}/save/recom_item_output/image_features_embedding.pkl", "wb"))
    pickle.dump(image_path_all, open(f"{root_path}/save/recom_item_output/candidate_img_files.pkl", "wb"))


def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices


def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0


def get_outfit(model, root_path, uploaded_file):
    transformers = get_recom_transform()
    candidates = os.listdir(f"{root_path}/data/recom_test/image/item")
    candidates = np.array(candidates)

    # display image
    show_images = Image.open(f"{root_path}/save/recom_input/{uploaded_file}")
    size = (400, 400)
    resized_im = show_images.resize(size)

    # to Tensor
    img_RGB = show_images.convert('RGB')

    # PIL to Tensor
    img_RGB_tensor_from_PIL = transformers['candidate_emb'](img_RGB)
    img_unsqueeze = torch.unsqueeze(img_RGB_tensor_from_PIL, 0).to(device)

    # extract features of uploaded image
    model.to(device)
    model.eval()
    with torch.no_grad():
        features = model(img_unsqueeze)

    # features into list
    features = torch.squeeze(features, 0).tolist()
    features_list = torch.load(f"{root_path}/save/recom_item_output/total.pt")
    # features_list = torch.Tensor([])
    # for emb_file in tqdm(os.listdir(f'{root_path}/save/recom_item_output/candidate_emb')):
    #     emb = torch.load(f'{root_path}/save/recom_item_output/candidate_emb/{emb_file}')
    #     features_list = torch.cat([features_list, emb], dim=0)

    img_indicess = recommend(features, features_list)
    similar = candidates[img_indicess].tolist()[0]
    styles, links = item_to_outfit(root_path, similar)
    return styles, links

# def visualize_model(model, dataloaders, class_names, num_images=6):
#     model.load_state_dict(torch.load('save/recom_model/model_recom.pt'))
#     was_training = model.training
#     model.eval()
#     images_so_far = 1
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             ax = plt.subplot(num_images // 2, 2, images_so_far)
#             ax.axis('off')
#             ax.set_title(labels.cpu()[i])
#             plt.imshow(inputs.cpu()[i].T)
#             for j in range(inputs.size()[0]-1):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 plt.imshow(inputs.cpu().data[j].T)
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#             plt.savefig(f'save/recom_item_output/{i}.png')
#             plt.close()
#         model.train(mode=was_training)
