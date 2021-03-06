import torch


class RecomConfig:
    classes = ["top", "skirt", "trouser", "outwear", "shoes", "hat", "accessory", "bag", "sneakers", "socks",
               "denim_pants", "women_bag", "sneakers", "socks", "training_pants"]
    NUM_CLASSES = len(classes)
    in_features = 512

    def __init__(self, root_path: str):
        self.train_data_dir = f"{root_path}/data/recom_train/image"
        self.test_item_dir = f"{root_path}/data/recom_test/image/item"
        self.test_outfit_dir = f"{root_path}/data/recom_test/image/style"


class MaskConfig:
    classes = ["top", "skirt", "trouser", "outwear"]

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
    num_epochs = 10

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __init__(self, root_path: str):
        self.musinsa_img_dir = f"{root_path}/data/mask_data/image"
        self.musinsa_json_dir = f"{root_path}/data/mask_data/total_dataset.json"
