import torch
import cv2
import random
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF

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


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(
                image, target)

        return image, target


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):
        w, h = image.size
        image = image.resize(self.size)

        _masks = target['masks'].copy()
        masks = np.zeros((_masks.shape[0], self.size[0], self.size[1]))

        for i, v in enumerate(_masks):
            v = Image.fromarray(v).resize(self.size, resample=Image.BILINEAR)
            masks[i] = np.array(v, dtype=np.uint8)

        target['masks'] = masks
        target['boxes'][:, [0, 2]] *= self.size[0] / w
        target['boxes'][:, [1, 3]] *= self.size[1] / h

        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
