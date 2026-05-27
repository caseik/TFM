import torch.nn as nn
from torchvision import models

from ._image_runner_utils import BaseImageClassificationRunner


class ViTB16Runner(BaseImageClassificationRunner):
    model_key = "vit_b16"
    img_size = 224

    def build_model(self, num_classes):
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
