import torch.nn as nn
from torchvision import models

from ._image_runner_utils import BaseImageClassificationRunner


class DenseNet121Runner(BaseImageClassificationRunner):
    model_key = "densenet121"
    img_size = 224

    def build_model(self, num_classes):
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model