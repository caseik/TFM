from dataset_manager import DatasetManager
from experiment_logger import ExperimentLogger
from split_manager import SplitManager
from feature_extractor import FeatureExtractor
from model_runners.mlp_runner import MLPRunner
from model_runners.centroid_runner import CentroidRunner
from metrics_generator import MetricsGenerator


def build_experiment_list(dataset):
    dataset_size = len(dataset)

    return [
   
        {
            "input_mode": "embeddings",
            "feature_extractor": "dinov2",
            "runner": "centroid",
            "train": 50, #int(dataset_size * 0.7),
            "val": int(dataset_size * 0.0),
            "test": "rest", #int(dataset_size * 0.15),
            "lora": int(dataset_size * 0.0),
            "augmentation": False
        },
        {
            "input_mode": "embeddings",
            "feature_extractor": "dinov2",
            "runner": "mlp",
            "train": 50, #int(dataset_size * 0.1),
            "val": int(dataset_size * 0),
            "test": "rest", #int(dataset_size * 0.15),
            "lora": int(dataset_size * 0),
            "augmentation": False
        },
        {
            "input_mode": "embeddings",
            "feature_extractor": "dinov2",
            "runner": "mlp",
            "train": 70, #int(dataset_size * 0.1),
            "val": int(dataset_size * 0),
            "test": "rest", #int(dataset_size * 0.15),
            "lora": int(dataset_size * 0),
            "augmentation": False
        }
    ]


def build_runner_registry():
    return {
        "mlp": MLPRunner(),
        "centroid": CentroidRunner()
    }


def run_experiments(
    dataset,
    experiments,
    split_manager,
    feature_extractor
):
    runners = build_runner_registry()

    for experiment in experiments:
        metrics_generator = MetricsGenerator()

        split_bundle = split_manager.prepare_split(
            dataset,
            experiment
        )

        split_bundle = feature_extractor.transform_split_bundle(
            split_bundle,
            experiment["feature_extractor"]
        )

        runner = runners[
            experiment["runner"]
        ]

        predictions = runner.run(
            split_bundle
        )

        metrics_generator.process_experiment(
            experiment,
            predictions
        )

        print(
            metrics_generator.performance_rows
        )

        print(
            metrics_generator.false_negative_rows
        )

def main():
    dataset_manager = DatasetManager()
    experiment_logger = ExperimentLogger()
    split_manager = SplitManager()
    feature_extractor = FeatureExtractor()

    dataset = dataset_manager.load_dataset()

    experiments = build_experiment_list(
        dataset
    )

    experiment_logger.register_batch(
        experiments
    )

    experiment_logger.export_table(
        "table_1.csv"
    )

    run_experiments(
        dataset,
        experiments,
        split_manager,
        feature_extractor
    )


if __name__ == "__main__":
    main()