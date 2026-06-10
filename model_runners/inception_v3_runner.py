import torch.nn as nn
from torchvision import models

from ._image_runner_utils import BaseImageClassificationRunner


class InceptionV3Runner(BaseImageClassificationRunner):
    model_key = "inception_v3"
    img_size = 299

    def build_model(self, num_classes):
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if model.AuxLogits is not None:
            aux_in_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

        return model