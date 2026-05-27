import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal = self.alpha[targets] * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class HAM10000ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, class_to_idx=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx or self._build_class_mapping(self.df)

    @staticmethod
    def _build_class_mapping(df):
        classes = sorted(df["dx"].unique())
        return {label: idx for idx, label in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = self.class_to_idx[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, int(label)


class EfficientNetB0Runner:
    """
    Runner compatible with the project architecture.

    Expected split_bundle:
        {
            "train": DataFrame with at least columns ["path", "dx"],
            "val":   optional DataFrame with columns ["path", "dx"],
            "test":  DataFrame with columns ["path", "dx"]
        }

    Returned predictions:
        {"y_true": [...], "y_pred": [...]}

    This class adapts the original EfficientNetB0_HAM10000 notebook by removing
    dataset downloading and train/val/test splitting. Those responsibilities stay
    in dataset_manager.py and split_manager.py, so all models share the same data
    protocol for a fair paper comparison.
    """

    def __init__(
        self,
        epochs=12,
        batch_size=32,
        num_workers=2,
        lr=1e-4,
        weight_decay=1e-4,
        img_size=224,
        seed=42,
        gamma=2.0,
        save_dir="checkpoints",
        model_name="efficientnet_b0",
        use_weighted_sampler=True,
        use_focal_loss=True,
        freeze_backbone=False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.img_size = img_size
        self.seed = seed
        self.gamma = gamma
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.use_weighted_sampler = use_weighted_sampler
        self.use_focal_loss = use_focal_loss
        self.freeze_backbone = freeze_backbone

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = []

    def run(self, split_bundle):
        self.set_seed(self.seed)
        self.validate_inputs(split_bundle)

        train_df = split_bundle["train"].copy().reset_index(drop=True)
        test_df = split_bundle["test"].copy().reset_index(drop=True)
        val_df = split_bundle.get("val")

        if val_df is None or len(val_df) == 0:
            val_df = test_df.copy().reset_index(drop=True)
            print("[EfficientNetB0Runner] No validation split found. Using test split for model selection. For paper results, prefer a real validation split.")
        else:
            val_df = val_df.copy().reset_index(drop=True)

        class_order = self.get_class_order(train_df, val_df, test_df)
        class_to_idx = {label: idx for idx, label in enumerate(class_order)}

        train_loader, val_loader, test_loader = self.build_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_to_idx=class_to_idx,
        )

        model = self.build_model(num_classes=len(class_order))
        criterion = self.build_criterion(train_df, class_to_idx)
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        best_path = self.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            class_order=class_order,
        )

        checkpoint = torch.load(best_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics, y_true, y_pred, _ = self.run_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            train=False,
        )

        print("\n[EfficientNetB0Runner] TEST")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}" if value == value else f"{key}: NaN")

        return self.package_predictions(y_true, y_pred)

    def validate_inputs(self, split_bundle):
        for split_name in ["train", "test"]:
            if split_name not in split_bundle:
                raise ValueError(f"Missing required split: {split_name}")
            required_columns = {"path", "dx"}
            missing = required_columns - set(split_bundle[split_name].columns)
            if missing:
                raise ValueError(f"Split '{split_name}' is missing columns: {missing}")
            if len(split_bundle[split_name]) == 0:
                raise ValueError(f"Split '{split_name}' is empty")

    def get_class_order(self, *dataframes):
        preferred_order = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]
        present = set()
        for df in dataframes:
            present.update(df["dx"].dropna().unique().tolist())

        ordered = [label for label in preferred_order if label in present]
        ordered += sorted(present - set(ordered))
        return ordered

    def build_transforms(self):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

        return train_transform, eval_transform

    def build_dataloaders(self, train_df, val_df, test_df, class_to_idx):
        train_transform, eval_transform = self.build_transforms()

        train_dataset = HAM10000ImageDataset(train_df, train_transform, class_to_idx)
        val_dataset = HAM10000ImageDataset(val_df, eval_transform, class_to_idx)
        test_dataset = HAM10000ImageDataset(test_df, eval_transform, class_to_idx)

        sampler = None
        shuffle = True

        if self.use_weighted_sampler:
            train_labels = train_df["dx"].map(class_to_idx).values
            class_counts = np.array([
                max((train_labels == i).sum(), 1)
                for i in range(len(class_to_idx))
            ])
            class_weights_sampler = 1.0 / class_counts
            sample_weights = torch.DoubleTensor(class_weights_sampler[train_labels])
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader, test_loader

    def build_criterion(self, train_df, class_to_idx):
        labels = train_df["dx"].map(class_to_idx).values
        classes = np.arange(len(class_to_idx))

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=labels,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        if self.use_focal_loss:
            return FocalLoss(alpha=class_weights, gamma=self.gamma)

        return nn.CrossEntropyLoss(weight=class_weights)

    def build_model(self, num_classes):
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)

        if self.freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model.to(self.device)

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, scheduler, class_order):
        best_macro_f1 = -1.0
        self.save_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.save_dir / "best_efficientnet_b0_ham10000.pth"

        for epoch in range(1, self.epochs + 1):
            print(f"\n[EfficientNetB0Runner] Epoch {epoch}/{self.epochs}")

            train_metrics, _, _, _ = self.run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                train=True,
            )

            val_metrics, _, _, _ = self.run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                train=False,
            )

            scheduler.step(val_metrics["macro_f1"])

            self.history.append({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            print(
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"train_macro_f1={train_metrics['macro_f1']:.4f}"
            )
            print(
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "class_order": class_order,
                    "img_size": self.img_size,
                    "history": self.history,
                }, best_path)
                print(f"[EfficientNetB0Runner] Best model saved: {best_path}")

        return best_path

    def run_one_epoch(self, model, loader, criterion=None, optimizer=None, train=False):
        model.train() if train else model.eval()

        losses = []
        all_labels = []
        all_preds = []
        all_probs = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                if criterion is not None:
                    loss = criterion(logits, labels)
                    losses.append(loss.item())

                if train:
                    loss.backward()
                    optimizer.step()

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        }

        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")

        return metrics, y_true, y_pred, y_prob

    def package_predictions(self, y_true, y_pred):
        return {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
