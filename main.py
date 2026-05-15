from dataset_manager import DatasetManager
from experiment_logger import ExperimentLogger


def build_experiment_list(dataset):
    dataset_size = len(dataset)

    experiments = [
        {
            "model": "dinov2_vitb14",
            "variant": "embeddings",
            "train": int(dataset_size * 0.7),
            "val": int(dataset_size * 0.15),
            "test": int(dataset_size * 0.0),
            "lora": int(dataset_size * 0.0),
            "augmentation": False
        },
        {
            "model": "dinov2_vitb14",
            "variant": "lora",
            "train": int(dataset_size * 0.7),
            "val": int(dataset_size * 0.15),
            "test": int(dataset_size * 0.15),
            "lora": int(dataset_size * 0.0),
            "augmentation": False
        }
    ]

    return experiments


def create_splits(dataset):
    return dataset


def run_experiments(experiments):
    pass


def main():
    dataset_manager = DatasetManager()
    experiment_logger = ExperimentLogger()

    dataset = dataset_manager.load_dataset()

    splits = create_splits(dataset)

    experiments = build_experiment_list(splits)

    experiment_logger.register_batch(experiments)

    experiment_logger.export_table("table_1.csv")

    run_experiments(experiments)


if __name__ == "__main__":
    main()