import torch
import numpy as np
import cv2
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# COCO dataset
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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


def tensor2img(tensor):
    tensor = 127.5 * (tensor[0].data.cpu().float().numpy() + 1.0)
    img = tensor.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def apply_mask(image, mask, labels, boxes, file_name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    alpha = 1
    beta = 0.6  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    _, _, w, h = mask.shape
    segmentation_map = np.zeros((w, h, 3), np.uint8)

    for n in range(mask.shape[0]):
        if labels[n] == 0:
            continue
        else:
            color = COLORS[random.randrange(0, len(COLORS))]
            segmentation_map[:, :, 0] = np.where(mask[n] > 0.5, COLORS[labels[n]][0], 0)
            segmentation_map[:, :, 1] = np.where(mask[n] > 0.5, COLORS[labels[n]][1], 0)
            segmentation_map[:, :, 2] = np.where(mask[n] > 0.5, COLORS[labels[n]][2], 0)
            image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, dtype=cv2.CV_8U)

        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[n][0], boxes[n][1], color=color, thickness=2)

        print(class_names[labels[n]])
        # put the label text above the objects
        cv2.putText(image, class_names[labels[n]], (boxes[n][0][0], boxes[n][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)
    # image save
    cv2.imwrite(f'save/{file_name}.png', image)
