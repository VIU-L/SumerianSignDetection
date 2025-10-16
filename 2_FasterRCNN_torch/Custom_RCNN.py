import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

def modify_fasterrcnn(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    """
    Modify a pre-trained Faster R-CNN to have a different number of output classes.
    
    Args:
        model: A pre-trained Faster R-CNN model.
        num_classes: Total number of classes including background.
        
    Returns:
        The same model with modified box_predictor.
    """
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    return model
