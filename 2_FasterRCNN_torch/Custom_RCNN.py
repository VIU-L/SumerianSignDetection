import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class TwoMLPHeadWithDropout(nn.Module):
    """
    Custom ROI box head with dropout layers added after each FC layer.
    """
    def __init__(self, in_channels: int, representation_size: int, dropout_p: float = 0.5):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dropout1(self.relu(self.fc6(x)))
        x = self.dropout2(self.relu(self.fc7(x)))
        return x


def modify_fasterrcnn(model: nn.Module, num_classes: int, dropout_p: float = 0.5) -> nn.Module:
    """
    Perform surgery on a Faster R-CNN model:
      - Replace the ROI head with one that has dropout.
      - Replace the box predictor for a new number of classes.
    """
    # Get input feature size for ROI head
    in_features = model.roi_heads.box_head.fc6.in_features
    representation_size = model.roi_heads.box_head.fc7.out_features

    # Replace the ROI head with dropout version
    model.roi_heads.box_head = TwoMLPHeadWithDropout(
        in_channels=in_features,
        representation_size=representation_size,
        dropout_p=dropout_p
    )

    # Replace the predictor with new class count
    in_features_pred = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_pred, num_classes)

    return model
