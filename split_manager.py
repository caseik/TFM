import random
import pandas as pd


class SplitManager:

    def __init__(self, seed=42):
        self.seed = seed

    def prepare_split(self, dataset, experiment_config):
        random.seed(self.seed)

        directives = {
            "train": experiment_config["train"],
            "val": experiment_config["val"],
            "test": experiment_config["test"]
        }

        if experiment_config["lora"]:
            directives["lora"] = experiment_config["lora"]

        subsets = self.select_subsets(dataset, directives)

        subsets = self.apply_augmentation(
            subsets,
            experiment_config["augmentation"]
        )

        return self.build_split_bundle(subsets)

    def select_subsets(self, dataset, directives):
        remaining = dataset.copy()
        subsets = {}

        for split_name, split_size in directives.items():

            if split_size == 0:
                subsets[split_name] = remaining.iloc[0:0].copy()
                continue

            if split_size == "rest":
                subsets[split_name] = remaining
                continue

            selected = []

            classes = sorted(
                remaining["dx"].unique()
            )

            per_class = split_size // len(classes)

            for class_name in classes:

                class_subset = remaining[
                    remaining["dx"] == class_name
                ]

                sampled = class_subset.sample(
                    n=min(per_class, len(class_subset)),
                    random_state=self.seed
                )

                selected.append(sampled)

            selected = pd.concat(selected)

            remaining = remaining.drop(
                selected.index
            )

            subsets[split_name] = selected

        return subsets

    def apply_augmentation(self, subsets, augmentation):
        if not augmentation:
            return subsets

        train = subsets["train"]

        augmented = train.copy()

        subsets["train"] = pd.concat(
            [train, augmented],
            ignore_index=True
        )

        return subsets

    def build_split_bundle(self, subsets):
        return subsets