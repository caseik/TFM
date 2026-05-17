import hashlib
import json
from pathlib import Path

import numpy as np
from backbones.dino_extractor import (
    DinoExtractor
)


class FeatureExtractor:

    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def transform_split_bundle(
        self,
        split_bundle,
        config
    ):
        if config == "none":
            return split_bundle

        extractor = self.load_backbone(
            config
        )

        transformed = {}

        for split_name, split_df in split_bundle.items():

            cache_key = self.build_cache_key(
                split_name,
                config
            )

            cache_path = (
                self.cache_dir /
                f"{cache_key}.npz"
            )

            if cache_path.exists():

                embeddings = self.load_embeddings(
                    cache_path
                )

            else:

                embeddings = extractor.encode(
                    split_df["image_path"].tolist()
                )

                self.save_embeddings(
                    cache_path,
                    embeddings
                )

            split_copy = split_df.copy()

            split_copy["embedding"] = list(
                embeddings
            )

            transformed[
                split_name
            ] = split_copy

        return transformed

    def load_backbone(self, config):

        if config == "none":
            return None
     
        if config == "dinov2":
            return DinoExtractor()

        raise ValueError(
            f"Unknown backbone: {config}"
        )

    def build_cache_key(
        self,
        split_name,
        config
    ):
        payload = {
            "split": split_name,
            "backbone": config
        }

        signature = json.dumps(
            payload,
            sort_keys=True
        )

        signature = hashlib.md5(
            signature.encode()
        ).hexdigest()

        return (
            f"{config}_"
            f"{split_name}_"
            f"{signature}"
        )

    def save_embeddings(
        self,
        path,
        embeddings
    ):
        np.savez(
            path,
            embeddings=embeddings
        )

    def load_embeddings(
        self,
        path
    ):
        data = np.load(
            path,
            allow_pickle=True
        )

        return data[
            "embeddings"
        ]