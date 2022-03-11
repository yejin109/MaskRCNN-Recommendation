import torch

# COCO
classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']

# Custom Dataset
# classes = (
#     'top', 'blouse', 't-shirt', 'Knitted fabri', 'shirt', 'bra top',
#     'hood', 'blue jeans', 'pants', 'skirt', 'leggings', 'jogger pants',
#     'coat', 'jacket', 'jumper', 'padding jacket', 'best', 'kadigan',
#     'zip up', 'dress', 'jumpsuit')

batch_size = 1
score_threshold = 0.965
hidden_layer = 256
lr = 1e-3
max_size = 800
num_epochs = 5


class DeepFashion2Config:
    NAME = "deepfashion2"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 13  # Background + category

    USE_MINI_MASK = True

    batch_size = 4
    score_threshold = 0.965
    hidden_layer = 256
    lr = 1e-2
    max_size = 800
    num_epochs = 2

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __init__(self, root_path: str):
        self.train_img_dir = f"{root_path}/data/DF2/train/image"
        self.train_json_path = f"{root_path}/data/DF2/train/train.json"
        self.valid_img_dir = f"{root_path}/data/DF2/validation/image"
        self.valid_json_path = f"{root_path}/data/DF2/validation/validation.json"
        self.log_path = f"{root_path}/save/log"
