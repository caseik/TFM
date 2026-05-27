import torch.nn as nn
from torchvision import models

from ._image_runner_utils import BaseImageClassificationRunner


class ResNet50Runner(BaseImageClassificationRunner):
    model_key = "resnet50"
    img_size = 224

    def build_model(self, num_classes):
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
