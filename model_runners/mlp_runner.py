import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class MLPRunner:

    def run(self, split_bundle):
        self.validate_inputs(split_bundle)

        x_train, y_train = self.prepare_split(
            split_bundle["train"]
        )

        x_test, y_test = self.prepare_split(
            split_bundle["test"]
        )

        model = self.train_model(
            x_train,
            y_train
        )

        predictions = self.predict(
            model,
            x_test
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
            np.array(
            split["embedding"].tolist()
            ),
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

    def train_model(self, x, y):
        model = MLP(
            x.shape[1],
            len(y.unique())
        )

        optimizer = torch.optim.Adam(
            model.parameters()
        )

        loss_fn = nn.CrossEntropyLoss()

        for _ in range(10):
            logits = model(x)

            loss = loss_fn(
                logits,
                y
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def predict(self, model, x):
        with torch.no_grad():
            return model(x).argmax(dim=1)

    def package_predictions(self, y_true, y_pred):
        return {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        }