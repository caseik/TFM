from dataset_manager import DatasetManager
from experiment_logger import ExperimentLogger
from split_manager import SplitManager
from feature_extractor import FeatureExtractor
from model_runners.mlp_runner import MLPRunner
from model_runners.centroid_runner import CentroidRunner
from model_runners.convnext_tiny_runner import ConvNeXtTinyRunner
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
            "augmentation": False,
        },
        {
            "input_mode": "embeddings",
            "feature_extractor": "dinov2",
            "runner": "mlp",
            "train": 50,
            "val": 0,
            "test": "rest",
            "lora": 0,
            "augmentation": False,
        },
        {
            "input_mode": "images",
            "feature_extractor": None,
            "runner": "convnext_tiny",
            "train": 50,
            "val": 10,
            "test": "rest",
            "lora": 0,
            "augmentation": False,
            "paradigm": "end_to_end_finetuning",
        },
    ]


def build_runner_registry():
    return {
        "mlp": MLPRunner(),
        "centroid": CentroidRunner(),
        "convnext_tiny": ConvNeXtTinyRunner(
            epochs=12,
            batch_size=16,
            learning_rate=1e-4,
            checkpoint_path="best_convnext_tiny_ham10000.pth",
        ),
    }


def run_experiments(dataset, experiments, split_manager, feature_extractor):
    runners = build_runner_registry()

    for experiment in experiments:
        metrics_generator = MetricsGenerator()

        split_bundle = split_manager.prepare_split(dataset, experiment)

        # Solo extraemos embeddings si el experimento los necesita.
        if experiment.get("input_mode") == "embeddings":
            split_bundle = feature_extractor.transform_split_bundle(
                split_bundle,
                experiment["feature_extractor"],
            )

        runner = runners[experiment["runner"]]
        predictions = runner.run(split_bundle)

        metrics_generator.process_experiment(experiment, predictions)

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

    run_experiments(dataset, experiments, split_manager, feature_extractor)


if __name__ == "__main__":
    main()
