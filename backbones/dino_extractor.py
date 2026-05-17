from transformers import (
    AutoImageProcessor,
    AutoModel
)

from PIL import Image

import torch


class DinoExtractor:

    def __init__(self):

        self.model_name = (
            "facebook/dinov2-small"
        )

        self.processor = (
            AutoImageProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                self.model_name
            )
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def encode(
        self,
        image_paths
    ):
        embeddings = []

        with torch.no_grad():

            for image_path in image_paths:

                image = Image.open(
                    image_path
                ).convert(
                    "RGB"
                )

                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )

                inputs = {
                    key: value.to(
                        self.device
                    )
                    for key, value in inputs.items()
                }

                outputs = self.model(
                    **inputs
                )

                embedding = (
                    outputs.pooler_output
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                embeddings.append(
                    embedding
                )

        return embeddings