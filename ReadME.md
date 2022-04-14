# Recommendation based on Instance Segmentation and Data Embedding
***


Overview
***
![제목](/md_src/img1.png){: width="50%" height="40%"}{: .align-center}

This service recommends outfit based on what kind of clothes user has. In the first step, users 
upload their own clothes such as shirt or outwear in their own closet. Then, this model can recognize 
clothes in the image and find out the most similar items in the database. Finally, this service searches for
the outfit which contains that similar items and offers a series of outfit and its link so that user can buy that outfit
if they want.


### Local Requirements
***
```
$ pip install numpy
$ pip install pytorch
$ pip install opencv-python
$ pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=pythonAPI
$ pip install streamlit
```

### How to Run
***
1. Fork this repository
2. Install several packages. 
   In this step, some might need to install Visual C++ 2017 Build Tools
3. Run recommneder.py with streamlit. 
   Ex. ```streamlit run recommender.py```
4. You can upload your own clothes and get recommendation among our pretrained dataset.


### Implementation
***
![제목](/md_src/video.gif){: width="50%" height="40%"}{: .align-center}


### File description
***

기본적으로 루트 디텍토리에서 파일을 실행하기 때문에 소스코드 내부에서 파일 경로들은 다 루트디렉토리 기준으로 
작성되어 있습니다.
 
#### data
- df: DeepFashion 데이터
- mask_data : 예전에 크롤링한 **사진 + 라벨 + 마스크** 데이터
- recom_test : 최근에 크롤링한 **사진 + 링크** 데이터
- recom_train : 예전에 크롤링한 **사진 + 라벨** (+ 마스크) 데이터

1. 사진

각 폴더의 하위 폴더에 image 폴더에 실제 사진이 저장되어 있습니다.<br>
이 때 코디북과 무신사 사진은 함께 저장되어 있습니다.<br>
단 **recom_test**의 경우에 단일 아이템 사진이 저장되어 있는 **item** 폴더와 
코디 사진이 저장된 **style** 폴더로 구분되어 있습니다.<br> 

2. 세부 데이터(라벨 마스크 링크 등등)

각 폴더에 json파일로 저장되어 있습니다.

#### jupyter

주피터 파일들 보관하는 폴더입니다.<br>
크롤링코드랑 함께 추가적인 설명이 필요한 경우 jupyter로 작성해서 보관하시면 될 것 같습니다!

#### dataset
- mask_dataset.py : 마스크 모델의 경우 데이터셋 클래스를 커스터마이징한 코드가 있습니다.(TODO: TF에서 필요한 소스 코드)
- recom_dataset.py : 사진 폴더알려주면 알아서 데이터셋이 만들어지는 구조라 데이터셋 세팅 관련 내용이 있습니다.

#### model
- mask_model.py : 예전에 마스크 모델에서 사용하던 모델 코드입니다. TF로 수정해야합니다.

추천 모델의 경우 ResNet18을 사용하기 때문에 별도의 모델 소스 코드는 없습니다.    

#### save
- mask_model : mask 모델 파라미터 파일 저장 폴더
- mask_output : mask 결과 확인하기 위해서 사진 저장할 때 사용하는 폴더
- recom_input : 추천모델 돌릴 때 필요한 배경 날리고 해당 영역만 자른 사진 폴더
- recom_item_output : 추천모델에서 비슷한 단일 아이템 사진이 저장된 폴더
- recom_outfit_output : 단일 아이템이 포함된 코디 사진이 저장된 폴더

일부 폴더는 깃에 올려보니 빈 폴더라 올라가있지 않아서 보이지 않은데 위와 같이 정리할 계획입니다.

#### util
- mask_post.py : 배경 지우기 & 사진 저장(미진행)
- mask_pre.py : Resize하고 Transformer compose와 같이 사진 전처리 함수와 COCO의 경우 데이터 셋으로 만들어주는 함수
- recom_post.py : 사진 시각화, 아이템 기준으로 검색하도록 만드는 데이터 정리는 구현 (TODO: 아이템 --> 코디 변환, 코디북과 무신사 비율 조정)
- recom_pre.py : 사진 전처리(Transformer) 함수 (TODO: 카테고리 라벨링 함수)


#### config.py

- RecomConfig 
- MaskConfig

중간중간에 학습시키거나 class 이름 등 상수값들 모아둔 파일입니다.<br>

각 환경에 필요한 함수나 루트 등 정리하는 클래스입니다.  

#### main

- main : 처음 사진 입력부터 최종 코디 추천까지 전체 프로세스가 정리된 코드입니다.
- main_mask : TODO: TF로 학습시키는 과정이 필요합니다.
- main_recom : model train 함수가 있습니다.