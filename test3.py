import os
import pandas as pd
import kagglehub

from transformers import AutoImageProcessor, AutoModel

import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# =========================
# DATASET
# =========================

def download_dataset():

    path = kagglehub.dataset_download(
        "kmader/skin-cancer-mnist-ham10000"
    )

    if "HAM10000_metadata.csv" in os.listdir(
        path
    ):

        print(
            f"Dataset listo en: {path}"
        )

        return path

    versions_path = os.path.join(
        path,
        "versions"
    )

    if os.path.exists(
        versions_path
    ):

        version = os.listdir(
            versions_path
        )[0]

        real_path = os.path.join(
            versions_path,
            version
        )

        print(
            f"Dataset listo en: {real_path}"
        )

        return real_path

    raise RuntimeError(
        "Dataset no encontrado"
    )


def load_metadata(
    base_path
):

    metadata_path = os.path.join(
        base_path,
        "HAM10000_metadata.csv"
    )

    df = pd.read_csv(
        metadata_path
    )

    print(
        "\nDistribución original:"
    )

    print(
        df["dx"].value_counts()
    )

    return df


def build_image_paths(
    base_path,
    df
):

    img_dir_1 = os.path.join(
        base_path,
        "HAM10000_images_part_1"
    )

    img_dir_2 = os.path.join(
        base_path,
        "HAM10000_images_part_2"
    )

    image_paths = {}

    for folder in [
        img_dir_1,
        img_dir_2
    ]:

        for img in os.listdir(
            folder
        ):

            image_id = img.split(
                "."
            )[0]

            image_paths[
                image_id
            ] = os.path.join(
                folder,
                img
            )

    df["path"] = df[
        "image_id"
    ].map(
        image_paths
    )

    print(
        "\nSin ruta:",
        df["path"].isna().sum()
    )

    return df


def train_test_split_rest(
    df,
    n_train=50
):

    train_list = []
    test_list = []

    for label, group in df.groupby(
        "dx"
    ):

        group = group.sample(
            frac=1,
            random_state=42
        )

        train = group.iloc[
            :n_train
        ]

        test = group.iloc[
            n_train:
        ]

        train_list.append(
            train
        )

        test_list.append(
            test
        )

    df_train = pd.concat(
        train_list
    ).reset_index(
        drop=True
    )

    df_test = pd.concat(
        test_list
    ).reset_index(
        drop=True
    )

    print(
        "\nTrain:"
    )

    print(
        df_train["dx"].value_counts()
    )

    print(
        "\nTest:"
    )

    print(
        df_test["dx"].value_counts()
    )

    return (
        df_train,
        df_test
    )


# =========================
# DINOv2
# =========================

def load_dinov2():

    model_name = (
        "facebook/dinov2-small"
    )

    processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(
        device
    )

    model.eval()

    print(
        f"\nModelo en {device}"
    )

    return (
        processor,
        model,
        device
    )


def load_image(
    path
):

    return Image.open(
        path
    ).convert(
        "RGB"
    )


def get_patch_embedding(
    image,
    processor,
    model,
    device
):

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(
        device
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # quitar CLS
    patches = outputs.last_hidden_state[
        :,
        1:,
        :
    ]

    # [1,256,768]
    # -> [16,16,768]

    patches = patches.reshape(
        16,
        16,
        384
    )

    return patches.cpu().numpy()


def generate_patch_embeddings(
    df,
    processor,
    model,
    device
):

    embeddings = []
    labels = []

    total = len(
        df
    )

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        if i % 100 == 0:

            print(
                f"{i}/{total}"
            )

        try:

            image = load_image(
                row["path"]
            )

            emb = get_patch_embedding(
                image,
                processor,
                model,
                device
            )

            embeddings.append(
                emb
            )

            labels.append(
                row["dx"]
            )

        except Exception as e:

            print(
                f"Error: {e}"
            )

    return (
        np.array(
            embeddings
        ),
        np.array(
            labels
        )
    )


# =========================
# CNN
# =========================

class PatchCNN(
    nn.Module
):

    def __init__(
        self,
        num_classes=7
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                384,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):

        # NHWC -> NCHW

        x = x.permute(
            0,
            3,
            1,
            2
        )

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


# =========================
# TRAIN
# =========================

def train_patch_model(
    emb_train,
    labels_train,
    epochs=100
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(
        labels_train
    )

    X_train = torch.tensor(
        emb_train,
        dtype=torch.float32
    ).to(
        device
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    ).to(
        device
    )

    model = PatchCNN(
        num_classes=len(
            encoder.classes_
        )
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(
        epochs
    ):

        model.train()

        logits = model(
            X_train
        )

        loss = criterion(
            logits,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch} | "
                f"Loss: {loss.item():.4f}"
            )

    return (
        model,
        encoder
    )


# =========================
# EVAL
# =========================

def evaluate_patch_model(
    model,
    encoder,
    emb_test,
    labels_test
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    X_test = torch.tensor(
        emb_test,
        dtype=torch.float32
    ).to(
        device
    )

    y_true = encoder.transform(
        labels_test
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            X_test
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        y_pred = preds.cpu().numpy()

    acc = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nAccuracy: "
        f"{acc:.4f}"
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    print(
        "\nMatriz de confusión:"
    )

    print(
        cm_df
    )

def evaluate_patch_model_2(
    model,
    encoder,
    emb_test,
    labels_test,
    batch_size=32
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    y_true = encoder.transform(
        labels_test
    )

    model.eval()

    all_preds = []

    with torch.no_grad():

        for i in range(
            0,
            len(emb_test),
            batch_size
        ):

            batch = emb_test[
                i:i + batch_size
            ]

            X_batch = torch.tensor(
                batch,
                dtype=torch.float32
            ).to(
                device
            )

            logits = model(
                X_batch
            )

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

    y_pred = np.array(
        all_preds
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nAccuracy: {acc:.4f}"
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    print(
        "\nMatriz de confusión:"
    )

    print(
        cm_df
    )

# =========================
# MAIN
# =========================

def main():

    base_path = download_dataset()

    df = load_metadata(
        base_path
    )

    df = build_image_paths(
        base_path,
        df
    )

    df_train, df_test = train_test_split_rest(
        df,
        n_train=50
    )

    processor, model, device = load_dinov2()

    print(
        "\nEmbeddings train..."
    )

    emb_train, labels_train = generate_patch_embeddings(
        df_train,
        processor,
        model,
        device
    )

    print(
        "\nEmbeddings test..."
    )

    emb_test, labels_test = generate_patch_embeddings(
        df_test,
        processor,
        model,
        device
    )

    model, encoder = train_patch_model(
        emb_train,
        labels_train,
        epochs=100
    )

    evaluate_patch_model_2(
        model,
        encoder,
        emb_test,
        labels_test,
        batch_size=32
    )


if __name__ == "__main__":
    main()