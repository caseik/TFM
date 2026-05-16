from dataset_manager import DatasetManager
from experiment_logger import ExperimentLogger
from split_manager import SplitManager


def build_experiment_list(dataset):
    dataset_size = len(dataset)

    return [
        {
            "input_mode": "images",
            "feature_extractor": "none",
            "runner": "resnet50",
            "train": int(dataset_size * 0.7),
            "val": int(dataset_size * 0),
            "test": int(dataset_size * 0.15),
            "lora": int(dataset_size * 0),
            "augmentation": False
        },
        {
            "input_mode": "embeddings",
            "feature_extractor": "dinov2_vitb14",
            "runner": "mlp",
            "train": int(dataset_size * 0.7),
            "val": int(dataset_size * 0.15),
            "test": int(dataset_size * 0.15),
            "lora": int(dataset_size * 0.15),
            "augmentation": False
        }
    ]


def run_experiments(dataset, experiments, split_manager):
    for experiment in experiments:

        split_bundle = split_manager.prepare_split(
            dataset,
            experiment
        )

        print(
            experiment["input_mode"],
            experiment["runner"],
            split_bundle.keys()
        )


def main():
    dataset_manager = DatasetManager()
    experiment_logger = ExperimentLogger()
    split_manager = SplitManager()

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
        split_manager
    )


if __name__ == "__main__":
    main()