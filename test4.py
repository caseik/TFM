import kagglehub
import os
import pandas as pd
from transformers import AutoImageProcessor, AutoModel
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import torch
import numpy as np


def evaluate_mlp(
    X_test,
    y_test,
    mlp_model,
    label_encoder,
    device
):

    mlp_model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        logits = mlp_model(
            X_tensor
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

    preds = preds.cpu().numpy()

    y_pred = label_encoder.inverse_transform(
        preds
    )

    # Accuracy
    acc = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nAccuracy: {acc:.4f}"
    )

    # Matriz de confusión
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=label_encoder.classes_
    )

    print(
        "\nMatriz de confusión:"
    )

    print(
        cm
    )

    # Opcional: métricas por clase
    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return acc, cm

class EmbeddingMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):
        return self.net(x)


def train_mlp(
    X_train,
    y_train,
    device,
    epochs=20,
    batch_size=16,
    lr=1e-3
):

    # Labels string -> int
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y_train
    )

    # Numpy -> Torch
    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_encoded,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    input_dim = X_train.shape[1]

    num_classes = len(
        encoder.classes_
    )

    model = EmbeddingMLP(
        input_dim,
        num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    model.train()

    for epoch in range(
        epochs
    ):

        epoch_loss = 0

        for x_batch, y_batch in loader:

            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                x_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"MLP Epoch {epoch+1}/{epochs} "
            f"Loss: {epoch_loss/len(loader):.4f}"
        )

    return model, encoder

def generate_embeddings(
    df,
    model,
    processor,
    device
):

    embeddings = []
    labels = []

    model.eval()

    with torch.no_grad():

        for _, row in df.iterrows():

            image = Image.open(
                row["path"]
            ).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            pixel_values = inputs[
                "pixel_values"
            ].to(device)

            outputs = model(
                pixel_values=pixel_values
            )

            embedding = outputs.last_hidden_state[
                :, 0
            ]

            embedding = embedding.squeeze(
                0
            ).cpu().numpy()

            embeddings.append(
                embedding
            )

            labels.append(
                row["dx"]
            )

    X = np.array(
        embeddings
    )

    y = np.array(
        labels
    )

    print(
        f"Embeddings generados: {X.shape}"
    )

    return X, y

def train_lora(
    model,
    processor,
    lora_df,
    device,
    epochs=15,
    batch_size=8,
    lr=1e-4
):

    # Labels
    classes = sorted(lora_df["dx"].unique())
    label_map = {
        label: idx
        for idx, label in enumerate(classes)
    }

    # Dataset
    dataset = HAMDataset(
        lora_df,
        processor,
        label_map
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Insertar LoRA en atención del transformer
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(
        model,
        lora_config
    )

    # Head temporal
    hidden_size = model.config.hidden_size

    classifier = nn.Linear(
        hidden_size,
        len(classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(classifier.parameters()),
        lr=lr
    )

    model.train()

    for epoch in range(epochs):

        epoch_loss = 0

        for pixel_values, labels in loader:

            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values
            )

            embeddings = outputs.last_hidden_state[:, 0]

            logits = classifier(
                embeddings
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss: {epoch_loss/len(loader):.4f}"
        )

    return model


class HAMDataset(Dataset):
    def __init__(self, df, processor, label_map):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        label = self.label_map[row["dx"]]

        return pixel_values, label
# =========================
# DATASET
# =========================

def download_dataset():
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

    # Caso 1: ya estamos dentro de la versión correcta
    if "HAM10000_metadata.csv" in os.listdir(path):
        print(f"Dataset listo en: {path}")
        return path

    # Caso 2: hay carpeta versions
    versions_path = os.path.join(path, "versions")

    if os.path.exists(versions_path):
        version = os.listdir(versions_path)[0]
        real_path = os.path.join(versions_path, version)
        print(f"Dataset listo en: {real_path}")
        return real_path

    # fallback (por si cambia estructura)
    raise RuntimeError(f"No se encontró estructura válida en: {path}")


def build_image_paths(base_path, df):
    img_dir_1 = os.path.join(base_path, "HAM10000_images_part_1")
    img_dir_2 = os.path.join(base_path, "HAM10000_images_part_2")

    image_paths = {}

    for folder in [img_dir_1, img_dir_2]:
        for img in os.listdir(folder):
            image_id = img.split(".")[0]
            image_paths[image_id] = os.path.join(folder, img)

    df["path"] = df["image_id"].map(image_paths)

    print("\nImágenes sin ruta:", df["path"].isna().sum())

    return df

def load_metadata(base_path):
    metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")
    df = pd.read_csv(metadata_path)

    print("\nDistribución original de clases:")
    print(df["dx"].value_counts())

    return df

def load_dinov2():
    #model_name = "facebook/dinov2-small"
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Modelo cargado en {device}")

    return processor, model, device


def split_for_lora_mlp_test(
    df,
    lora_per_class=30,
    mlp_per_class=50,
    random_state=42
):
    lora_rows = []
    mlp_rows = []
    test_rows = []

    for dx in sorted(df["dx"].unique()):

        # Todas las imágenes de esa clase
        class_df = df[df["dx"] == dx].sample(
            frac=1,
            random_state=random_state
        ).reset_index(drop=True)

        total_needed = lora_per_class + mlp_per_class

        if len(class_df) < total_needed:
            raise ValueError(
                f"La clase {dx} solo tiene {len(class_df)} imágenes, "
                f"pero necesitas {total_needed}"
            )

        # Split sin solapamiento
        lora_split = class_df.iloc[:lora_per_class]
        mlp_split = class_df.iloc[
            lora_per_class:lora_per_class + mlp_per_class
        ]
        test_split = class_df.iloc[
            lora_per_class + mlp_per_class:
        ]

        lora_rows.append(lora_split)
        mlp_rows.append(mlp_split)
        test_rows.append(test_split)

    lora_df = pd.concat(lora_rows).reset_index(drop=True)
    mlp_df = pd.concat(mlp_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    print("\nLoRA:")
    print(lora_df["dx"].value_counts())

    print("\nMLP:")
    print(mlp_df["dx"].value_counts())

    print("\nTest:")
    print(test_df["dx"].value_counts())

    return lora_df, mlp_df, test_df
# =========================
# MAIN
# =========================

def main():
    # Dataset
    base_path = download_dataset()
    df = load_metadata(base_path)
    df = build_image_paths(base_path, df)

    # Modelo
    processor, model, device = load_dinov2()
    lora_df, mlp_df, test_df = split_for_lora_mlp_test(df)

    lora_model = train_lora(
    model=model,
    processor=processor,
    lora_df=lora_df,
    device=device
    )
    X_train, y_train = generate_embeddings(
        mlp_df,
        lora_model,
        processor,
        device
    )
    
    X_test, y_test = generate_embeddings(
        test_df,
        lora_model,
        processor,
        device
    )
    mlp_model, label_encoder = train_mlp(
        X_train,
        y_train,
        device
    )

    acc, cm = evaluate_mlp(
        X_test,
        y_test,
        mlp_model,
        label_encoder,
        device
    )

if __name__ == "__main__":
    main()