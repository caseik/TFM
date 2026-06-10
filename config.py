from dataclasses import dataclass
from typing import Any

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

SEEDS = [42, 1337, 2026]

DEFAULT_SCENARIO = "few_shot_50"

SCENARIOS: dict[str, dict[str, Any]] = {
    # Cuenta aproximada por clase/lesión para train y val.
    # El test queda fijo con el resto de lesiones no vistas.
    "few_shot_50": {
        "description": "Pocos datos: 50 lesiones por clase para entrenamiento y 10 para validación.",
        "train": 50,
        "val": 10,
        "test": "rest",
        "group_by": "lesion_id",
    },
    "few_shot_100": {
        "description": "Pocos datos ampliado: 100 lesiones por clase para entrenamiento y 20 para validación.",
        "train": 100,
        "val": 20,
        "test": "rest",
        "group_by": "lesion_id",
    },
    "stratified_70_15_15": {
        "description": "Partición estratificada clásica por lesión: 70 % train, 15 % val, 15 % test.",
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
        "group_by": "lesion_id",
    },
}

EMBEDDING_EXPERIMENTS = [
    {
        "input_mode": "embeddings",
        "feature_extractor": "dinov2",
        "runner": "centroid",
        "augmentation": False,
    },
    {
        "input_mode": "embeddings",
        "feature_extractor": "dinov2",
        "runner": "mlp",
        "augmentation": False,
    },
]

IMAGE_EXPERIMENTS = [
    {
        "input_mode": "images",
        "feature_extractor": "none",
        "runner": "efficientnet_b0",
        "augmentation": True,
    },
    {
        "input_mode": "images",
        "feature_extractor": "none",
        "runner": "convnext_tiny",
        "augmentation": True,
    },
    {
        "input_mode": "images",
        "feature_extractor": "none",
        "runner": "mobilenet_v2",
        "augmentation": True,
    },
    {
        "input_mode": "images",
        "feature_extractor": "none",
        "runner": "vit_b16",
        "augmentation": True,
    },
    {
        "input_mode": "images",
        "feature_extractor": "none",
        "runner": "resnet50",
        "augmentation": True,
    },
    {
        "input_mode": "images",
        "feature_extractor": "none",
        "runner": "densenet121",
        "augmentation": True,
    },
    {
        "input_mode": "images",
        "feature_extractor": "none",
        "runner": "inception_v3",
        "augmentation": True,
    },
]
