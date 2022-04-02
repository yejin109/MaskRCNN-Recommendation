import streamlit as st

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model.recom_model import ResNet_without_fc
from util.recom_pre import get_recom_transform
from util.recom_post import save_file, get_outfit
from main_mask import test_mask_model
from config import RecomConfig, MaskConfig
from model.mask_model import get_mask_model
from model.mask_model import EmbeddingExtractor
from util.mask_post import tensor2img, apply_mask

root_path = os.getcwd()
config = RecomConfig(root_path)
recom_in_features = config.in_features
num_classes = config.NUM_CLASSES

device = torch.device('cpu')


def show(borad, idx):
    with borad:
        st.header(f"{idx}")
        image = Image.open(f"{outfit_path}/{styles[idx][0]}.jpg")
        st.image(image)
        st.write(f"{links[idx]}")


# 마스크 모델
mask_config = MaskConfig(root_path)
hidden_layer = mask_config.hidden_layer
json_path = mask_config.musinsa_json_dir
image_dir_path = mask_config.musinsa_img_dir
classes = mask_config.classes
max_size = mask_config.max_size
mask_model = get_mask_model(num_classes, hidden_layer)
mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
mask_in_features = mask_model.roi_heads.box_predictor.cls_score.in_features
box_predictor_param = mask_model.roi_heads.box_predictor.state_dict()
embedding_extractor = EmbeddingExtractor(mask_in_features, num_classes, box_predictor_param)
mask_model.roi_heads.box_predictor = embedding_extractor
mask_model.load_state_dict(torch.load('save/mask_model/model_mask.pt'))
mask_model.to(device)
mask_model.eval()

# 추천 모델
resnet_wo_fc = ResNet_without_fc([2, 2, 2, 2], recom_in_features, num_classes, True).to(device)
resnet_wo_fc.load_state_dict(torch.load('save/recom_model/pure.pt'))
# resnet_wo_fc.load_state_dict(torch.load('save/recom_model/model_recom.pt'))
transformers = get_recom_transform()
resnet_wo_fc.to(device)
resnet_wo_fc.eval()


outfit_path = f"{root_path}/data/recom_test/image/style"
####################

# show_images = Image.open('uploader/다운로드.jpg')
#
# image = mask_transform(show_images).unsqueeze(dim=0).to(device)
# image = image[:, :3, :, :]
#
# result = mask_model(image)
#
# image = tensor2img(image[0])
# scores = list(result[0]['scores'].detach().cpu().numpy())
#
# thresholded_preds_inidices = [np.argmax(scores)]
# thresholded_preds_count = len(thresholded_preds_inidices)
#
# mask = result[0]['masks']
# mask = mask[:thresholded_preds_count]
#
# labels = result[0]['labels']
#
# boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes']]
# boxes = boxes[:thresholded_preds_count]
#
# mask = mask.data.float().cpu().numpy()
# recom_input_name = apply_mask(image, mask, labels, boxes, 'test', classes)
#
# similar, styles, links = get_outfit(resnet_wo_fc, root_path, recom_input_name)
#
# styles = [i for i in styles]
# links = [j for j in links]


st.title('Clothing recommender system')
uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        show_images = Image.open(uploaded_file)
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)
        input_id = 'test'
        image = mask_transform(show_images).unsqueeze(dim=0).to(device)
        image = image[:, :3, :, :]
        print(image.size())
        result = mask_model(image)

        image = tensor2img(image[0])
        scores = list(result[0]['scores'].detach().cpu().numpy())

        thresholded_preds_inidices = [np.argmax(scores)]
        thresholded_preds_count = len(thresholded_preds_inidices)

        mask = result[0]['masks']
        mask = mask[:thresholded_preds_count]

        labels = result[0]['labels']

        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes']]
        boxes = boxes[:thresholded_preds_count]

        mask = mask.data.float().cpu().numpy()
        recom_input_name = apply_mask(image, mask, labels, boxes, input_id, classes)

        show_images = Image.open(f"{root_path}/save/mask_output/{input_id}.png")
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)

        similar, styles, links = get_outfit(resnet_wo_fc, root_path, recom_input_name)

        styles = [i for i in styles]
        links = [j for j in links]

        print("="*30)
        print(similar)
        print(styles)
        print("="*30)

        count = 0
        for i in range(10):
            col1, col2, = st.columns(2)
            with col1:
                st.header(f"Source")
                # image = Image.open(f"{root_path}/data/mask_data/image/{similar[count]}")
                image = Image.open(f"{root_path}/data/recom_test/image/item/{similar[count]}")
                st.image(image)
            with col2:
                st.header(f"Outfit")
                image = Image.open(f"{outfit_path}/style_{styles[count]}.jpg")
                st.image(image)
                st.write(f"{links[count][0]}")
            count += 1

    else:
        st.header("Some error occur")
