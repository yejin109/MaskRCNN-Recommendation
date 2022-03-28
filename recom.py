import os
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2
from tqdm import tqdm
from torchvision.models import resnet50
from config import Configuration
from dataset import RecomDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor
import streamlit as st
from PIL import Image
from util.utils import save_file

root_path = os.getcwd()
file_names = os.listdir('data/sample/')

config = Configuration(root_path)
max_size = config.max_size

transform = Compose([
    Resize((max_size, max_size)),
    ToTensor(),
    ])
dataset = RecomDataset(file_names[:100], img_path=f'{root_path}/data/sample', device=config.device, transforms=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = resnet50(pretrained=True).to(device=config.device)
model.eval()

embedding_list = []
for img in tqdm(dataloader):
    result_to_resnet = model(img)
    flatten_result = result_to_resnet.flatten().detach().cpu().numpy()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)
    embedding_list.append(result_normlized)

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(embedding_list)

distence, indices = neighbors.kneighbors([result_normlized])

print(indices)

for file in indices[0][1:6]:
    tmp_img = cv2.imread(f"{root_path}/data/sample/{file_names[file]}")
    tmp_img = cv2.resize(tmp_img, (200,200))
    cv2.imshow("output", tmp_img)
    cv2.waitKey(0)

# uploaded_file = st.file_uploader("Choose your image")
# if uploaded_file is not None:
#     if save_file(uploaded_file):
#         # display image
#         show_images = Image.open(uploaded_file)
#         size = (max_size, max_size)
#         resized_im = show_images.resize(size)
#         st.image(resized_im)
#
#         # extract features of uploaded image
#         features = model(os.path.join("uploader", uploaded_file.name))
#         #st.text(features)
#         img_indicess = neighbors.kneighbors(features, embedding_list)
#         col1,col2,col3,col4,col5 = st.columns(5)
#
#         with col1:
#             st.header("I")
#             st.image(file_names[img_indicess[0][0]])
#
#         with col2:
#             st.header("II")
#             st.image(file_names[img_indicess[0][1]])
#
#         with col3:
#             st.header("III")
#             st.image(file_names[img_indicess[0][2]])
#
#         with col4:
#             st.header("IV")
#             st.image(file_names[img_indicess[0][3]])
#
#         with col5:
#             st.header("V")
#             st.image(file_names[img_indicess[0][4]])
#     else:
#         st.header("Some error occur")