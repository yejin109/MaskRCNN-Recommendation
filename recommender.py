import streamlit as st

import pandas as pd
import numpy as np
from numpy.linalg import norm

import os
import pickle
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.models.resnet as resnet

# 미리 정의
conv1x1=resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock= resnet.BasicBlock

features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

class ResNet_without_fc(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet_without_fc, self).__init__()
        self.inplanes = 64

        # inputs = 3x224x224 -> 3x128x128로 바뀜
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # 마찬가지로 전부 사이즈 조정
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0]) # 3 반복
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 4 반복
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 6 반복
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 3 반복
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1): # planes -> 입력되는 채널 수
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input [32, 128, 128] -> [C ,H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #x.shape =[32, 64, 64]

        x = self.layer1(x)
        #x.shape =[128, 64, 64]
        x = self.layer2(x)
        #x.shape =[256, 32, 32]
        x = self.layer3(x)
        #x.shape =[512, 16, 16]
        x = self.layer4(x)
        #x.shape =[1024, 8, 8]
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

device = torch.device('cpu')
feature_model_one = ResNet_without_fc(resnet.BasicBlock, [2, 2, 2, 2], 2, True)
feature_model_one.load_state_dict(torch.load('no_fc_model_state_dict.pt', map_location=device))

# torchvision.transforms
tf_resize = transforms.Resize((224,224))
tf_toTensor = transforms.ToTensor()


st.title('Clothing recommender system')

def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        # display image
        show_images = Image.open(uploaded_file)
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)

        # to Tensor
        img_RGB  = Image.open(uploaded_file).convert('RGB')
        # PIL to Tensor
        img_RGB_tensor_from_PIL = tf_toTensor(tf_resize(img_RGB))
        img_unsqueeze = torch.unsqueeze(img_RGB_tensor_from_PIL, 0)

        # extract features of uploaded image
        feature_model_one.eval()
        with torch.no_grad():
            features=feature_model_one(img_unsqueeze)

        # features into list
        features = torch.squeeze(features,0).tolist()
        features_list = features_list.tolist()

        img_indicess = recommendd(features, features_list)
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.header("I")
            st.image(img_files_list[img_indicess[0][0]])

        with col2:
            st.header("II")
            st.image(img_files_list[img_indicess[0][1]])

        with col3:
            st.header("III")
            st.image(img_files_list[img_indicess[0][2]])

        with col4:
            st.header("IV")
            st.image(img_files_list[img_indicess[0][3]])

        with col5:
            st.header("V")
            st.image(img_files_list[img_indicess[0][4]])
    else:
        st.header("Some error occur")