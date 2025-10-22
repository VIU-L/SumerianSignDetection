import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image
import json
import os
import random
import torch.nn.functional as F
class CuneiformDataset(Dataset):
    def __init__(self, annotation_path, label_key="charname_id", transforms=None):
        """
        Args:
            annotation_path: path to your JSON file
            label_key: choose "charname_id", "transliteration_id", or "is_char"
            transforms: torchvision transforms for data augmentation
        """
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

        self.ids = list(self.annotations.keys())
        self.label_key = label_key
        self.transforms = transforms if transforms else T.Compose([
            T.ConvertImageDtype(torch.float32)
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        data = self.annotations[img_id]

        img_path = data["image_path"]

        # Read image as a tensor (C, H, W)
        img = read_image(img_path).to(torch.float32)
        # if RGBA, convert to RGB
        if img.shape[0] == 4:
            img = img[:3, :, :]

        boxes, labels, areas = [], [], []
        for ann in data["bboxes"]:
            # bbox format in annotation: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = ann["bbox"]
            boxes.append([xmin, ymin, xmax, ymax])

            if self.label_key == "is_char":
                labels.append(1)
            else:
                label = ann[self.label_key]
                labels.append(label + 1)  # class indices must start at 1

            areas.append((xmax - xmin) * (ymax - ymin))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # data augmentation.
        img, boxes = crop_image_and_bbox(img, boxes, (10, 10, 10, 10), random_pick=True)
        img, boxes = change_ratio_image_and_bbox(img, boxes, ratio_range=(0.9, 1.1))
        # random blur
        if random.random() < 0.5:
            kernel_size = random.choice([3, 5])
            img = T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 1.2))(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def crop_image_and_bbox(image,boxes,crop_to,random_pick=True):
    """Crops the image and adjusts the bounding boxes accordingly.

    Args:
        image (Tensor): The input image tensor of shape (C, H, W).
        boxes (Tensor): The bounding boxes tensor of shape (N, 4) in (xmin, ymin, xmax, ymax) format.
        crop_to (tuple): The desired cropped out pixels, (crop_top,crop_bottom, crop_left,crop_right).
        random_pick (bool): If True, randomly pick crop values smaller than crop_to.
    Returns:
        cropped_image (Tensor): The cropped image tensor.
        cropped_boxes (Tensor): The adjusted bounding boxes tensor.
    """
    
    _, H, W = image.shape
    
    crop_top, crop_bottom, crop_left, crop_right = crop_to
    if random_pick:
        
        crop_top = random.randint(0, crop_top)
        crop_bottom = random.randint(0, crop_bottom)
        crop_left = random.randint(0, crop_left)
        crop_right = random.randint(0, crop_right)
        
    cropped_image = image[:, crop_top:H - crop_bottom, crop_left:W - crop_right]
    cropped_boxes = boxes.clone()
    cropped_boxes[:, 0] = boxes[:, 0] - crop_left  # xmin
    cropped_boxes[:, 1] = boxes[:, 1] - crop_top   # ymin
    cropped_boxes[:, 2] = boxes[:, 2] - crop_left  # xmax
    cropped_boxes[:, 3] = boxes[:, 3] - crop_top   # ymax
    # Ensure boxes are within the cropped image boundaries
    # cropped_boxes[:, 0] = torch.clamp(cropped_boxes[:, 0], min=0, max=W - crop_left - crop_right)  # xmin
    # cropped_boxes[:, 1] = torch.clamp(cropped_boxes[:, 1], min=0, max=H - crop_top - crop_bottom)    # ymin
    # cropped_boxes[:, 2] = torch.clamp(cropped_boxes[:, 2], min=0, max=W - crop_left - crop_right)  # xmax
    # cropped_boxes[:, 3] = torch.clamp(cropped_boxes[:, 3], min=0, max=H - crop_top - crop_bottom)    # ymax
    return cropped_image, cropped_boxes

def change_ratio_image_and_bbox(image,boxes,ratio_range=(0.9,1.1)):
    """Change the aspect ratio of the image and adjust the bounding boxes accordingly.

    Args:
        image (Tensor): The input image tensor of shape (C, H, W).
        boxes (Tensor): The bounding boxes tensor of shape (N, 4) in (xmin, ymin, xmax, ymax) format.
        ratio_range (tuple): The range of aspect ratio change to choose from (min_ratio, max_ratio).
    Returns:
        Tensor: The resized image tensor.
        Tensor: The adjusted bounding boxes tensor.
    """
    ratio = random.uniform(*ratio_range)
    _, H, W = image.shape
    new_H = int(H * ratio)
    new_W = int(W / ratio)
    
    resized_image = F.interpolate(image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0] * new_W / W  # xmin
    boxes[:, 1] = boxes[:, 1] * new_H / H  # ymin
    boxes[:, 2] = boxes[:, 2] * new_W / W  # xmax
    boxes[:, 3] = boxes[:, 3] * new_H / H  # ymax
    return resized_image, boxes