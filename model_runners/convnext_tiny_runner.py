import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms


class HAM10000ImageDataset(Dataset):
    """Dataset compatible con los DataFrame generados por SplitManager.

    Espera columnas:
    - path: ruta de imagen JPG
    - dx: etiqueta textual original de HAM10000

    También acepta image_path como alternativa a path.
    """

    def __init__(self, dataframe, class_to_idx, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform

        if "path" not in self.df.columns and "image_path" in self.df.columns:
            self.df["path"] = self.df["image_path"]

        if "path" not in self.df.columns:
            raise ValueError("ConvNeXtTinyRunner necesita una columna 'path' o 'image_path'.")

        if "dx" not in self.df.columns:
            raise ValueError("ConvNeXtTinyRunner necesita una columna 'dx' con la clase.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = self.class_to_idx[row["dx"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


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


class ConvNeXtTinyRunner:
    """Runner end-to-end para ConvNeXt-Tiny.

    Encaja con la interfaz común del proyecto:
        predictions = runner.run(split_bundle)

    Devuelve:
        {"y_true": [...], "y_pred": [...]}

    Diferencia metodológica:
    - No consume embeddings.
    - Consume imágenes directamente desde la columna path.
    - Hace fine-tuning supervisado end-to-end.
    """

    def __init__(
        self,
        img_size=224,
        batch_size=16,
        epochs=12,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gamma=2.0,
        num_workers=2,
        seed=42,
        checkpoint_path="best_convnext_tiny_ham10000.pth",
        use_weighted_sampler=True,
        pretrained=True,
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.num_workers = num_workers
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.use_weighted_sampler = use_weighted_sampler
        self.pretrained = pretrained

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_order = None
        self.class_to_idx = None
        self.idx_to_class = None

        self.set_seed(self.seed)

    def run(self, split_bundle):
        self.validate_inputs(split_bundle)
        self.prepare_label_mapping(split_bundle)

        train_loader, val_loader, test_loader, class_weights = self.build_dataloaders(split_bundle)
        model = self.build_model(num_classes=len(self.class_order))
        criterion = FocalLoss(alpha=class_weights, gamma=self.gamma)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        best_macro_f1 = -1.0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            print(f"\n===== ConvNeXt-Tiny | Epoch {epoch}/{self.epochs} =====")

            train_metrics, _, _ = self.run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                train=True,
            )

            if val_loader is not None:
                val_metrics, _, _ = self.run_one_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    optimizer=None,
                    train=False,
                )
                monitor_value = val_metrics["macro_f1"]
                scheduler.step(monitor_value)
            else:
                val_metrics = None
                monitor_value = train_metrics["macro_f1"]

            self.print_epoch_metrics(train_metrics, val_metrics)

            if monitor_value > best_macro_f1:
                best_macro_f1 = monitor_value
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "class_order": self.class_order,
                    "img_size": self.img_size,
                }
                torch.save(best_state, self.checkpoint_path)
                print(f"Mejor modelo guardado en: {self.checkpoint_path}")

        if best_state is None and Path(self.checkpoint_path).exists():
            best_state = torch.load(self.checkpoint_path, map_location=self.device)

        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])

        _, y_true, y_pred = self.run_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            train=False,
        )

        return self.package_predictions(y_true, y_pred)

    def validate_inputs(self, split_bundle):
        if "train" not in split_bundle or "test" not in split_bundle:
            raise ValueError("split_bundle debe incluir al menos 'train' y 'test'.")

        if len(split_bundle["train"]) == 0:
            raise ValueError("El split de entrenamiento está vacío.")

        if len(split_bundle["test"]) == 0:
            raise ValueError("El split de test está vacío.")

    def prepare_label_mapping(self, split_bundle):
        preferred_order = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]
        present_classes = sorted(set(split_bundle["train"]["dx"]).union(set(split_bundle["test"]["dx"])))

        if set(present_classes).issubset(set(preferred_order)):
            self.class_order = [c for c in preferred_order if c in present_classes]
        else:
            self.class_order = present_classes

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_order)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

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

    def build_dataloaders(self, split_bundle):
        train_transform, eval_transform = self.build_transforms()

        train_dataset = HAM10000ImageDataset(
            split_bundle["train"],
            class_to_idx=self.class_to_idx,
            transform=train_transform,
        )

        test_dataset = HAM10000ImageDataset(
            split_bundle["test"],
            class_to_idx=self.class_to_idx,
            transform=eval_transform,
        )

        val_loader = None
        if "val" in split_bundle and len(split_bundle["val"]) > 0:
            val_dataset = HAM10000ImageDataset(
                split_bundle["val"],
                class_to_idx=self.class_to_idx,
                transform=eval_transform,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        train_labels = split_bundle["train"]["dx"].map(self.class_to_idx).to_numpy()

        class_weights_loss = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(self.class_order)),
            y=train_labels,
        )
        class_weights_loss = torch.tensor(
            class_weights_loss,
            dtype=torch.float32,
            device=self.device,
        )

        if self.use_weighted_sampler:
            class_sample_count = np.array([
                np.sum(train_labels == class_id)
                for class_id in range(len(self.class_order))
            ])
            class_sample_count = np.maximum(class_sample_count, 1)
            class_weights_sampler = 1.0 / class_sample_count
            sample_weights = class_weights_sampler[train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
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

        return train_loader, val_loader, test_loader, class_weights_loss

    def build_model(self, num_classes):
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if self.pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model.to(self.device)

    def run_one_epoch(self, model, loader, criterion=None, optimizer=None, train=False):
        model.train() if train else model.eval()

        losses = []
        all_labels = []
        all_preds = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                if criterion is not None:
                    loss = criterion(logits, labels)
                    losses.append(loss.item())

                if train:
                    loss.backward()
                    optimizer.step()

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

        metrics = {
            "loss": float(np.mean(losses)) if losses else None,
            "accuracy": accuracy_score(all_labels, all_preds),
            "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        }

        return metrics, all_labels, all_preds

    def print_epoch_metrics(self, train_metrics, val_metrics=None):
        print(
            "TRAIN | "
            f"loss={train_metrics['loss']:.4f} | "
            f"acc={train_metrics['accuracy']:.4f} | "
            f"bal_acc={train_metrics['balanced_accuracy']:.4f} | "
            f"macro_f1={train_metrics['macro_f1']:.4f}"
        )

        if val_metrics is not None:
            print(
                "VAL   | "
                f"loss={val_metrics['loss']:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} | "
                f"bal_acc={val_metrics['balanced_accuracy']:.4f} | "
                f"macro_f1={val_metrics['macro_f1']:.4f}"
            )

    def package_predictions(self, y_true, y_pred):
        return {
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
