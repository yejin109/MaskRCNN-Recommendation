import streamlit as st

import os
import pickle
from PIL import Image

import torch

from model.recom_model import ResNet_without_fc
from util.recom_pre import get_recom_transform
from util.recom_post import save_file, recommendd
from config import RecomConfig

features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

root_path = os.getcwd()
config = RecomConfig(root_path)
in_features = config.in_features
num_classes = config.NUM_CLASSES

device = torch.device('cpu')
resnet_wo_fc = ResNet_without_fc([2, 2, 2, 2], in_features, num_classes, True).to(device)
resnet_wo_fc.load_state_dict(torch.load('save/recom_model/model_recom.pt'))
transformers = get_recom_transform()

# to Tensor
img_RGB = Image.open(uploaded_file).convert('RGB')
# PIL to Tensor
img_RGB_tensor_from_PIL = transformers['emb'](img_RGB)
img_unsqueeze = torch.unsqueeze(img_RGB_tensor_from_PIL, 0)

# extract features of uploaded image
resnet_wo_fc.eval()
with torch.no_grad():
    features = resnet_wo_fc(img_unsqueeze)

# features into list
features = torch.squeeze(features, 0).tolist()
features_list = features_list.tolist()

img_indicess = recommendd(features, features_list)

# st.title('Clothing recommender system')
# uploaded_file = st.file_uploader("Choose your image")
#
# if uploaded_file is not None:
#     if save_file(uploaded_file):
#         # display image
#         show_images = Image.open(uploaded_file)
#         size = (400, 400)
#         resized_im = show_images.resize(size)
#         st.image(resized_im)
#
#         # to Tensor
#         img_RGB = Image.open(uploaded_file).convert('RGB')
#         # PIL to Tensor
#         img_RGB_tensor_from_PIL = transformers['emb'](img_RGB)
#         img_unsqueeze = torch.unsqueeze(img_RGB_tensor_from_PIL, 0)
#
#         # extract features of uploaded image
#         resnet_wo_fc.eval()
#         with torch.no_grad():
#             features = resnet_wo_fc(img_unsqueeze)
#
#         # features into list
#         features = torch.squeeze(features, 0).tolist()
#         features_list = features_list.tolist()
#
#         img_indicess = recommendd(features, features_list)
#         col1, col2, col3, col4, col5 = st.columns(5)
#
#         with col1:
#             st.header("I")
#             st.image(img_files_list[img_indicess[0][0]])
#
#         with col2:
#             st.header("II")
#             st.image(img_files_list[img_indicess[0][1]])
#
#         with col3:
#             st.header("III")
#             st.image(img_files_list[img_indicess[0][2]])
#
#         with col4:
#             st.header("IV")
#             st.image(img_files_list[img_indicess[0][3]])
#
#         with col5:
#             st.header("V")
#             st.image(img_files_list[img_indicess[0][4]])
#     else:
#         st.header("Some error occur")
