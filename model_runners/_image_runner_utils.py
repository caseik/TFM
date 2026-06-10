from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight


HAM10000_CLASS_ORDER = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HAM10000ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, class_order=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.class_order = class_order or HAM10000_CLASS_ORDER
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_order)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row.get("path", row.get("image_path", None))
        if image_path is None:
            raise ValueError("The dataframe must include a 'path' or 'image_path' column")

        image = Image.open(image_path).convert("RGB")

        if "class_idx" in row:
            label = int(row["class_idx"])
        elif "label" in row:
            label = int(row["label"])
        elif "dx" in row:
            label = self.label_to_idx[str(row["dx"])]
        else:
            raise ValueError("The dataframe must include 'class_idx', 'label' or 'dx'")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BaseImageClassificationRunner:
    model_key = "base_image_model"
    img_size = 224

    def __init__(
        self,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_workers=2,
        gamma=2.0,
        patience=5,
        save_dir="checkpoints",
        seed=42
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.gamma = gamma
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_order = HAM10000_CLASS_ORDER

    def run(self, split_bundle):
        self.validate_inputs(split_bundle)
        self.set_seed()

        train_df = split_bundle["train"].reset_index(drop=True)
        val_df = split_bundle.get("val")
        test_df = split_bundle["test"].reset_index(drop=True)

        if val_df is None or len(val_df) == 0:
            val_df = test_df.copy()
        else:
            val_df = val_df.reset_index(drop=True)

        train_loader, val_loader, test_loader = self.build_dataloaders(
            train_df,
            val_df,
            test_df
        )

        model = self.build_model(num_classes=len(self.class_order)).to(self.device)
        criterion = self.build_criterion(train_df)
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        model = self.train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer
        )

        y_true, y_pred = self.predict(model, test_loader)
        return self.package_predictions(y_true, y_pred)

    def validate_inputs(self, split_bundle):
        if "train" not in split_bundle or "test" not in split_bundle:
            raise ValueError("split_bundle must include at least 'train' and 'test'")
        for split_name in ["train", "test"]:
            split = split_bundle[split_name]
            if "path" not in split.columns and "image_path" not in split.columns:
                raise ValueError(f"Split '{split_name}' must include a path or image_path column")

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def build_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.10
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        return train_transform, eval_transform

    def get_labels(self, df):
        if "class_idx" in df.columns:
            return df["class_idx"].astype(int).values
        if "label" in df.columns:
            return df["label"].astype(int).values
        if "dx" in df.columns:
            mapping = {label: idx for idx, label in enumerate(self.class_order)}
            return df["dx"].map(mapping).astype(int).values
        raise ValueError("DataFrame must include class_idx, label or dx")

    def build_dataloaders(self, train_df, val_df, test_df):
        train_transform, eval_transform = self.build_transforms()

        train_dataset = HAM10000ImageDataset(
            train_df,
            transform=train_transform,
            class_order=self.class_order
        )
        val_dataset = HAM10000ImageDataset(
            val_df,
            transform=eval_transform,
            class_order=self.class_order
        )
        test_dataset = HAM10000ImageDataset(
            test_df,
            transform=eval_transform,
            class_order=self.class_order
        )

        train_labels = self.get_labels(train_df)
        class_counts = np.array([
            max((train_labels == i).sum(), 1)
            for i in range(len(self.class_order))
        ])
        class_weights_sampler = 1.0 / class_counts
        sample_weights = torch.DoubleTensor(class_weights_sampler[train_labels])

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def build_criterion(self, train_df):
        train_labels = self.get_labels(train_df)
        present_classes = np.unique(train_labels)

        weights = np.ones(len(self.class_order), dtype=np.float32)
        if len(present_classes) > 1:
            computed = compute_class_weight(
                class_weight="balanced",
                classes=present_classes,
                y=train_labels
            )
            for cls, weight in zip(present_classes, computed):
                weights[int(cls)] = float(weight)

        weights_tensor = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device
        )

        return FocalLoss(alpha=weights_tensor, gamma=self.gamma)

    def train_model(self, model, train_loader, val_loader, criterion, optimizer):
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)

            print(
                f"[{self.model_key}] Epoch {epoch + 1:02d}/{self.epochs} "
                f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
                self.save_checkpoint(model, optimizer, epoch, best_val_loss)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print(f"[{self.model_key}] Early stopping")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def train_one_epoch(self, model, loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            logits = self.extract_logits(outputs)
            loss = criterion(logits, targets)

            if hasattr(outputs, "aux_logits") and outputs.aux_logits is not None:
                loss = loss + 0.4 * criterion(outputs.aux_logits, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return running_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def evaluate(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            outputs = model(images)
            logits = self.extract_logits(outputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return running_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()
        y_true = []
        y_pred = []

        for images, targets in loader:
            images = images.to(self.device)
            outputs = model(images)
            logits = self.extract_logits(outputs)
            preds = logits.argmax(dim=1).detach().cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(targets.tolist())

        return y_true, y_pred

    def extract_logits(self, outputs):
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs

    def save_checkpoint(self, model, optimizer, epoch, val_loss):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{self.model_key}_best.pth"
        torch.save({
            "model_key": self.model_key,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_order": self.class_order,
            "img_size": self.img_size,
        }, path)

    def package_predictions(self, y_true, y_pred):
        return {
            "y_true": list(map(int, y_true)),
            "y_pred": list(map(int, y_pred))
        }

    def build_model(self, num_classes):
        raise NotImplementedError