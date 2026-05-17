import torch


class CentroidRunner:

    def run(self, split_bundle):
        self.validate_inputs(split_bundle)

        x_train, y_train = self.prepare_split(
            split_bundle["train"]
        )

        x_test, y_test = self.prepare_split(
            split_bundle["test"]
        )

        centroids = self.compute_centroids(
            x_train,
            y_train
        )

        predictions = self.predict(
            x_test,
            centroids
        )

        return self.package_predictions(
            y_test,
            predictions
        )

    def validate_inputs(self, split_bundle):
        if "embedding" not in split_bundle["train"]:
            raise ValueError("Embeddings required")

    def prepare_split(self, split):
        x = torch.tensor(
            split["embedding"].tolist(),
            dtype=torch.float32
        )

        classes = sorted(
            split["dx"].unique()
        )

        mapping = {
            label: idx
            for idx, label in enumerate(classes)
        }

        y = torch.tensor(
            split["dx"].map(mapping).tolist(),
            dtype=torch.long
        )

        return x, y

    def compute_centroids(self, x, y):
        centroids = []

        for class_id in y.unique():
            class_vectors = x[
                y == class_id
            ]

            centroids.append(
                class_vectors.mean(dim=0)
            )

        return torch.stack(
            centroids
        )

    def predict(self, x, centroids):
        distances = torch.cdist(
            x,
            centroids
        )

        return distances.argmin(
            dim=1
        )

    def package_predictions(self, y_true, y_pred):
        return {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        }