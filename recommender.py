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
in_features = config.in_features
num_classes = config.NUM_CLASSES

device = torch.device('cpu')


# 마스크 모델
mask_config = MaskConfig(root_path)
hidden_layer = mask_config.hidden_layer
json_path = mask_config.musinsa_json_dir
image_dir_path = mask_config.musinsa_img_dir
classes = mask_config.classes
max_size = mask_config.max_size
mask_model = get_mask_model(num_classes, hidden_layer)
mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

mask_model.eval()

# 추천 모델
resnet_wo_fc = ResNet_without_fc([2, 2, 2, 2], in_features, num_classes, True).to(device)
resnet_wo_fc.load_state_dict(torch.load('save/recom_model/model_recom.pt'))
transformers = get_recom_transform()

resnet_wo_fc.eval()

# st.title('Clothing recommender system')
uploaded_file = st.file_uploader("Choose your image")
outfit_path = f"{root_path}/data/recom_test/image/style"

in_features = mask_model.roi_heads.box_predictor.cls_score.in_features
box_predictor_param = mask_model.roi_heads.box_predictor.state_dict()
embedding_extractor = EmbeddingExtractor(in_features, num_classes, box_predictor_param)
mask_model.roi_heads.box_predictor = embedding_extractor
mask_model.load_state_dict(torch.load('save/mask_model/model_mask.pt'))
show_images = Image.open('uploader/다운로드.jpg')
image = mask_transform(show_images)

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
apply_mask(image, mask, labels, boxes, 0, classes)

styles, links = get_outfit(resnet_wo_fc, root_path, uploaded_file)

styles = [i for i in styles]
links = [j for j in links]
if uploaded_file is not None:
    if save_file(uploaded_file):
        in_features = mask_model.roi_heads.box_predictor.cls_score.in_features
        box_predictor_param = mask_model.roi_heads.box_predictor.state_dict()
        embedding_extractor = EmbeddingExtractor(in_features, num_classes, box_predictor_param)
        mask_model.roi_heads.box_predictor = embedding_extractor
        mask_model.load_state_dict(torch.load('save/mask_model/model_mask.pt'))
    #     show_images = Image.open(uploaded_file)
    #     image = mask_transform(show_images)
    #
    #     result = mask_model(image)
    #
    #     image = tensor2img(image[0])
    #     scores = list(result[0]['scores'].detach().cpu().numpy())
    #
    #     thresholded_preds_inidices = [np.argmax(scores)]
    #     thresholded_preds_count = len(thresholded_preds_inidices)
    #
    #     mask = result[0]['masks']
    #     mask = mask[:thresholded_preds_count]
    #
    #     labels = result[0]['labels']
    #
    #     boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes']]
    #     boxes = boxes[:thresholded_preds_count]
    #
    #     mask = mask.data.float().cpu().numpy()
    #     apply_mask(image, mask, labels, boxes, 0, classes)
    #
    #     styles, links = get_outfit(resnet_wo_fc, root_path, uploaded_file)
    #
    #     styles = [i for i in styles]
    #     links = [j for j in links]
    #
    #     col1, col2, col3, col4, col5 = st.columns(5)
    #     with col1:
    #         st.header("I")
    #         st.image(f"{outfit_path}/{styles[0]}")
    #
    #     with col2:
    #         st.header("II")
    #         st.image(f"{outfit_path}/{styles[1]}")
    #
    #     with col3:
    #         st.header("III")
    #         st.image(f"{outfit_path}/{styles[2]}")
    #
    #     with col4:
    #         st.header("IV")
    #         st.image(f"{outfit_path}/{styles[3]}")
    #
    #     with col5:
    #         st.header("V")
    #         st.image(f"{outfit_path}/{styles[4]}")
    # else:
    #     st.header("Some error occur")
