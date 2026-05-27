from dataset_manager import DatasetManager
from experiment_logger import ExperimentLogger
from split_manager import SplitManager
from feature_extractor import FeatureExtractor

from model_runners.mlp_runner import MLPRunner
from model_runners.centroid_runner import CentroidRunner
from model_runners.efficientnet_b0_runner import EfficientNetB0Runner
from model_runners.convnext_tiny_runner import ConvNeXtTinyRunner
from model_runners.mobilenet_v2_runner import MobileNetV2Runner
from model_runners.vit_b16_runner import ViTB16Runner
from model_runners.resnet50_runner import ResNet50Runner
from model_runners.densenet121_runner import DenseNet121Runner
from model_runners.inception_v3_runner import InceptionV3Runner

from metrics_generator import MetricsGenerator


def build_experiment_list(dataset):
    return [
        {
            "input_mode": "embeddings",
            "feature_extractor": "dinov2",
            "runner": "centroid",
            "train": 50,
            "val": 0,
            "test": "rest",
            "lora": 0,
            "augmentation": False
        },
        {
            "input_mode": "embeddings",
            "feature_extractor": "dinov2",
            "runner": "mlp",
            "train": 50,
            "val": 0,
            "test": "rest",
            "lora": 0,
            "augmentation": False
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "efficientnet_b0",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": True
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "convnext_tiny",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": True
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "mobilenet_v2",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": True
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "vit_b16",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": True
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "resnet50",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": True
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "densenet121",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": True
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "inception_v3",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": True
        }
    ]


def build_runner_registry():
    return {
        "mlp": MLPRunner(),
        "centroid": CentroidRunner(),
        "efficientnet_b0": EfficientNetB0Runner(),
        "convnext_tiny": ConvNeXtTinyRunner(),
        "mobilenet_v2": MobileNetV2Runner(),
        "vit_b16": ViTB16Runner(),
        "resnet50": ResNet50Runner(),
        "densenet121": DenseNet121Runner(),
        "inception_v3": InceptionV3Runner(),
    }


def run_experiments(dataset, experiments, split_manager, feature_extractor):
    runners = build_runner_registry()

    for experiment in experiments:
        metrics_generator = MetricsGenerator()

        split_bundle = split_manager.prepare_split(dataset, experiment)

        if experiment.get("input_mode") == "embeddings":
            split_bundle = feature_extractor.transform_split_bundle(
                split_bundle,
                experiment["feature_extractor"]
            )

        runner = runners[experiment["runner"]]
        predictions = runner.run(split_bundle)

        metrics_generator.process_experiment(
            experiment,
            predictions
        )

        print(metrics_generator.performance_rows)
        print(metrics_generator.false_negative_rows)


def main():
    dataset_manager = DatasetManager()
    experiment_logger = ExperimentLogger()
    split_manager = SplitManager()
    feature_extractor = FeatureExtractor()

    dataset = dataset_manager.load_dataset()
    experiments = build_experiment_list(dataset)

    experiment_logger.register_batch(experiments)
    experiment_logger.export_table("table_1.csv")

    run_experiments(
        dataset,
        experiments,
        split_manager,
        feature_extractor
    )


if __name__ == "__main__":
    main()
