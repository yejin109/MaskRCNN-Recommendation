import os
import numpy as np

from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Crawling Dataset
class ImgDataset(Dataset):
    def __init__(self, max_size, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        files = []
        file_list = os.listdir('data/')
        for file in file_list:
            # 이 때 Img의 길이를 일치시켜야 하는 문제 존재
            img = Image.open(f'data/{file}').resize((max_size, max_size), Image.LANCZOS)
            files.append(self.transform(img))

        self.files = files

    def __getitem__(self, index):
        return self.files[index].to(device)

    def __len__(self):
        return len(self.files)


# DACON - Pytorch Dataset
class FashionDataset(Dataset):
    def __init__(self, root_path, image_path, device, transforms=None):
        self.coco = COCO(root_path)
        self.image_ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms
        self.image_path = image_path
        self.device= device

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        file_name = f'{self.image_path}/{file_name}'
        image = Image.open(file_name).convert('RGB')

        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = [x for x in self.coco.loadAnns(annot_ids) if x['image_id'] == image_id]

        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32)
        masks = np.array([self.coco.annToMask(annot) for annot in annots], dtype=np.uint8)

        area = np.array([annot['area'] for annot in annots], dtype=np.float32)
        iscrowd = np.array([annot['iscrowd'] for annot in annots], dtype=np.uint8)

        target = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32).to(device)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8).to(device)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64).to(device)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32).to(device)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8).to(device)

        image.to(device)
        return image, target


# DeepFashion2
class DeepFashion2Dataset(Dataset):
    def load_coco(self, image_dir, json_path, class_ids=None,
                  class_map=None, return_coco=False):
        """Load the DeepFashion2 dataset.
        """

        coco = COCO(json_path)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("deepfashion2", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "deepfashion2", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_keypoint(self, image_id):
        """
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2Dataset, self).load_mask(image_id)

        instance_keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                keypoint = annotation['keypoints']

                instance_keypoints.append(keypoint)
                class_ids.append(class_id)

        keypoints = np.stack(instance_keypoints, axis=1)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2Dataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(DeepFashion2Dataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        super(DeepFashion2Dataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m