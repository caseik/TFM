import torch.nn as nn
from torchvision import models

from ._image_runner_utils import BaseImageClassificationRunner


class MobileNetV2Runner(BaseImageClassificationRunner):
    model_key = "mobilenet_v2"
    img_size = 224

    def build_model(self, num_classes):
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
