import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from PIL import Image

import pandas as pd


class SkinDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        label = int(
            row["class_idx"]
        )

        return image, label


class ResNet50Runner:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def print_split_info(self, split_bundle):

        print("\n===== TRAIN SPLIT =====")

        train_counts = (
            split_bundle["train"]["dx"]
            .value_counts()
            .sort_index()
        )

        for label, count in train_counts.items():

            print(f"{label}: {count}")

        print(f"TOTAL TRAIN: {len(split_bundle['train'])}")

        print("\n===== TEST SPLIT =====")

        test_counts = (
            split_bundle["test"]["dx"]
            .value_counts()
            .sort_index()
        )

        for label, count in test_counts.items():

            print(f"{label}: {count}")

        print(f"TOTAL TEST: {len(split_bundle['test'])}")

    def run(self, split_bundle):

        self.print_split_info(split_bundle)
        train_loader, test_loader = (
            self.prepare_dataloaders(
                split_bundle
            )
        )

        num_classes = len(
            split_bundle["train"]["class_idx"].unique()
        )

        model = self.build_model(
            num_classes
        )

        model = self.train_model(
            model,
            train_loader
        )

        predictions, labels = self.predict(
            model,
            test_loader
        )

        return self.package_predictions(
            labels,
            predictions
        )

    def prepare_dataloaders(
        self,
        split_bundle
    ):

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_dataset = SkinDataset(
            split_bundle["train"],
            transform=train_transform
        )

        test_dataset = SkinDataset(
            split_bundle["test"],
            transform=test_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False
        )

        return train_loader, test_loader

    def build_model(
        self,
        num_classes
    ):

        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2
        )

        model = models.resnet50(
            weights=weights
        )

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

        return model.to(
            self.device
        )

    def train_model(
        self,
        model,
        train_loader
    ):


        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

        loss_fn = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(5):

            epoch_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                logits = model(
                    images
                )

                loss = loss_fn(
                    logits,
                    labels
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

                preds = logits.argmax(
                    dim=1
                )

                correct += (
                    preds == labels
                ).sum().item()

                total += len(labels)

            avg_loss = (
                epoch_loss / len(train_loader)
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
        test_loader
    ):

        model.eval()

        predictions = []

        labels = []

        with torch.no_grad():

            for images, y in test_loader:

                images = images.to(
                    self.device
                )

                logits = model(
                    images
                )

                preds = logits.argmax(
                    dim=1
                )

                predictions.extend(
                    preds.cpu().tolist()
                )

                labels.extend(
                    y.tolist()
                )

        return predictions, labels

    def package_predictions(
        self,
        y_true,
        y_pred
    ):

        return {
            "y_true": y_true,
            "y_pred": y_pred
        }