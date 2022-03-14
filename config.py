import torch


class DeepFashion2Config:
    # DF2
    classes = [
        "rshort sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling", "shorts",
        "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress"
        ]

    # Project
    # classes = ["top", "trouser", "outwear", "skirt"]

    NAME = "deepfashion2"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + len(classes)  # Background + category

    USE_MINI_MASK = True

    batch_size = 4
    score_threshold = 0.8
    hidden_layer = 128
    lr = 1e-2
    weight_decay = 1e-3
    max_size = 800
    num_epochs = 5

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __init__(self, root_path: str):
        self.train_img_dir = f"{root_path}/data/DF2/train/image"
        self.train_json_path = f"{root_path}/data/DF2/train/train.json"
        self.valid_img_dir = f"{root_path}/data/DF2/validation/image"
        self.valid_json_path = f"{root_path}/data/DF2/validation/validation.json"
        self.log_path = f"{root_path}/save/log"
