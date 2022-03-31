import torch
import cv2
import random
import numpy as np


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def tensor2img(tensor):
    tensor = 127.5 * (tensor.data.cpu().float().numpy() + 1.0)
    img = tensor.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def apply_mask(image, masks, labels, boxes, file_name, classes):
    labels = labels -1
    final_labels = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # width = image.shape[0]
    height = image.shape[1]

    alpha = 1
    beta = 0.6  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    _, _, w, h = masks.shape
    segmentation_map = np.zeros((w, h, 3), np.uint8)

    for n in range(masks.shape[0]):
        box = boxes[n]
        trunc_image = image[box[0][1]:box[1][1], box[0][0]:box[1][0], :]
        cv2.imwrite(f'save/trunc_png/{file_name}_{classes[labels[n]]}.png', trunc_image)

        color = COLORS[random.randrange(0, len(COLORS))]
        segmentation_map[:, :, 0] = np.where(masks[n] > 0.5, COLORS[labels[n]][0], 0)
        segmentation_map[:, :, 1] = np.where(masks[n] > 0.5, COLORS[labels[n]][1], 0)
        segmentation_map[:, :, 2] = np.where(masks[n] > 0.5, COLORS[labels[n]][2], 0)
        image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, dtype=cv2.CV_8U)

        # draw the bounding boxes around the objects
        # cv2.rectangle(image, boxes[n][0], boxes[n][1], color=color, thickness=2)

        # put the label text above the objects
        cv2.putText(image, classes[labels[n]], (boxes[n][0][0], boxes[n][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)
        final_labels.append(labels[n].detach().cpu().numpy())

    # image save
    cv2.imwrite(f'save/png/{file_name}.png', image)
    return final_labels
