# extract predicted bounding boxes from output, and plot on the image
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
def plot_predictions(img_input,output_dict_of_this_image,thres=0.8,category_dict=None):
    """
    img_input: input image as a tensor (C, H, W)
    output_dict_of_this_image: output dictionary from the model for this image
    thres: threshold for displaying boxes based on score
    """
    img=img_input.clone()
    labels=output_dict_of_this_image['labels']
    boxes=output_dict_of_this_image['boxes']
    scores=output_dict_of_this_image['scores']
    label_texts=[category_dict[i] for i in labels] 
    threshold=thres

    for label, box, score,label_text in zip(labels, boxes, scores,label_texts):
        if score >= threshold:
            print(box, label_text, score)
            # Prepare box and label for draw_bounding_boxes
            box_tensor = box.unsqueeze(0).int()
            label_str = [label_text + f": {score:.2f}"]
            img = draw_bounding_boxes(img, box_tensor, labels=label_str, colors="red", width=2, font_size=20)
    plt.figure(figsize=(12, 8))
    plt.imshow(img.permute(1, 2, 0).cpu().numpy()) # plt uses (H, W, C)
    plt.axis('off')
    plt.show()
    
def run_and_plot(image_path,model,transforms,category_dict=None,thres=0.8,device="cuda"):
    """
    image_path: path to the image file
    model: trained object detection model
    transforms: transformations to apply to the image
    category_dict: dictionary mapping label indices to category names
    thres: threshold for displaying boxes based on score
    """
    img = read_image(image_path)
        # if RGBA, convert to RGB
    if img.shape[0] == 4:
        print("Converting RGBA to RGB")
        img = img[:3, :, :]
    if transforms:
        img = transforms(img).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img.unsqueeze(0))[0]  # model expects a list of images
    if not category_dict:
        category_dict = model.category_dict
    plot_predictions(img, output, thres, category_dict)
    return output
    