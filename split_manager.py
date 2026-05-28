import random
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import os
import numpy as np
from PIL import Image





class SplitManager:



    def generate_augmented_samples(self, samples):



        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        output_dir = "augmented"

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        augmented_rows = []

        for index, row in samples.iterrows():

            image_path = row["path"]

            image = load_img(
                image_path
            )

            image_array = img_to_array(
                image
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            augmented_batch = next(
                datagen.flow(
                    image_array,
                    batch_size=1
                )
            )

            augmented_image = (augmented_batch[0].astype("uint8"))

            filename = (f"aug_{index}_{self.seed}.jpg")

            augmented_path = os.path.join(output_dir, filename)

            Image.fromarray(augmented_image).save(augmented_path)

            augmented_row = row.copy()

            augmented_row["path"] = (augmented_path)

            augmented_rows.append(augmented_row)

        return pd.DataFrame(augmented_rows)
    

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

        subsets = self.select_subsets(
            dataset,
            directives
        )

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

                subsets[split_name] = (
                    remaining.iloc[0:0].copy()
                )

                continue

            if split_size == "rest":

                subsets[split_name] = (
                    remaining.copy()
                )

                continue

            selected = []

            classes = sorted(
                remaining["dx"].unique()
            )

            for class_name in classes:

                class_subset = remaining[
                remaining["dx"] == class_name
            ]

                class_size = len(
                    class_subset
                )
            

                # FEW SHOT
                if split_size >= 1:

                    per_class = int(
                        split_size
                    )

                # PERCENTAGE
                else:

                    per_class = int(
                        class_size * split_size
                    )

                sampled = class_subset.sample(
                    n=min(
                        per_class,
                        class_size
                    ),
                    random_state=self.seed
                )

                selected.append(
                    sampled
                )

            selected = pd.concat(
                selected
            )

            remaining = remaining.drop(
                selected.index
            )

            subsets[split_name] = (
                selected
            )

        return subsets

    def apply_augmentation(self, subsets, augmentation):

        if not augmentation:
            return subsets

        MIN_SAMPLES = 500
        MAX_SAMPLES = 1200

        train = subsets["train"]

        balanced = []

        classes = sorted(
            train["dx"].unique()
        )

        for class_name in classes:

            class_subset = train[
                train["dx"] == class_name
            ]

            class_size = len(
                class_subset
            )

          # UNDERSAMPLING
            if class_size > MAX_SAMPLES:

                selected_subset = class_subset.sample(
                    n=MAX_SAMPLES,
                    random_state=self.seed
                )

                discarded_subset = class_subset.drop(
                    selected_subset.index
                )

                subsets["test"] = pd.concat(
                    [

                        subsets["test"],
                        discarded_subset
                    ],
                    ignore_index=True
                    )

                class_subset = selected_subset

            # OVERSAMPLING / DUMMY AUGMENTATION
            elif class_size < MIN_SAMPLES:

                missing = (
                    MIN_SAMPLES
                    - class_size
                )

                # Duplicate rows so the generator
                # can apply different augmentations
                # on each epoch / batch
                augmented = class_subset.sample(
                    n=missing,
                    replace=True,
                    random_state=self.seed
                ).copy()

                augmented = self.generate_augmented_samples( augmented )

                class_subset = pd.concat(
                    [class_subset, augmented],
                    ignore_index=True
                )


            balanced.append(
                class_subset
            )

        subsets["train"] = pd.concat(
            balanced,
            ignore_index=True
        )

        return subsets

    def build_split_bundle(
        self,
        subsets
    ):

        return subsets