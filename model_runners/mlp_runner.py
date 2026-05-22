import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


class MLP(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):
        return self.layers(x)


class MLPRunner:

    def run(self, split_bundle):

        self.validate_inputs(
            split_bundle
        )

        x_train, y_train = self.prepare_split(
            split_bundle["train"]
        )

        x_test, y_test = self.prepare_split(
            split_bundle["test"]
        )

        print("\n===== TRAIN INFO =====")

        print(
            f"Train samples: {len(x_train)}"
        )

        unique, counts = np.unique(
            y_train.numpy(),
            return_counts=True
        )

        print("Class distribution:")

        for cls, count in zip(unique, counts):

            print(
                f"Class {cls}: {count}"
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

    def validate_inputs(
        self,
        split_bundle
    ):

        if "embedding" not in split_bundle["train"]:

            raise ValueError(
                "Embeddings required"
            )

    def prepare_split(
        self,
        split
    ):

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

    def train_model(
        self,
        x,
        y
    ):

        model = MLP(
            x.shape[1],
            len(y.unique())
        )

        model.train()

        dataset = TensorDataset(
            x,
            y
        )

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(15):

            epoch_loss = 0
            correct = 0
            total = 0

            for x_batch, y_batch in loader:

                logits = model(
                    x_batch
                )

                loss = loss_fn(
                    logits,
                    y_batch
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

                preds = logits.argmax(
                    dim=1
                )

                correct += (
                    preds == y_batch
                ).sum().item()

                total += len(y_batch)

            avg_loss = (
                epoch_loss / len(loader)
            )

            acc = correct / total

            print(
                f"Epoch {epoch+1:02d} "
                f"| Loss: {avg_loss:.4f} "
                f"| Train Acc: {acc:.4f}"
            )

        return model

    def predict(
        self,
        model,
        x
    ):

        model.eval()

        with torch.no_grad():

            return model(x).argmax(
                dim=1
            )

    def package_predictions(
        self,
        y_true,
        y_pred
    ):

        return {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        }