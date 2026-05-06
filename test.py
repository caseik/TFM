import os
import pandas as pd
import kagglehub
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA


def apply_pca_95(
    emb_train,
    emb_test
):
    # Ajustar SOLO en train
    pca = PCA(
        n_components=0.95
    )

    emb_train_pca = pca.fit_transform(
        emb_train
    )

    emb_test_pca = pca.transform(
        emb_test
    )

    print(
        f"\nDim original: "
        f"{emb_train.shape[1]}"
    )

    print(
        f"Dim reducida PCA: "
        f"{emb_train_pca.shape[1]}"
    )

    return (
        emb_train_pca,
        emb_test_pca,
        pca
    )

def analyze_pca_variance(emb_train):
    
    pca = PCA()
    
    pca.fit(emb_train)

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    print("\nVarianza acumulada:")

    checkpoints = [
        10,
        20,
        30,
        50,
        75,
        100,
        150,
        200,
        emb_train.shape[1]
    ]

    for n in checkpoints:

        if n <= len(cumulative_variance):

            variance = cumulative_variance[n - 1]

            print(
                f"{n} componentes -> "
                f"{variance:.4f}"
            )

    # cuántos componentes para 95%
    n_95 = np.argmax(
        cumulative_variance >= 0.95
    ) + 1

    print(
        f"\nComponentes para 95% "
        f"de varianza: {n_95}"
    )

    return pca

def evaluate_mlp(
    model,
    encoder,
    emb_test,
    labels_test
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_test = torch.tensor(
        emb_test,
        dtype=torch.float32
    ).to(device)

    y_true = encoder.transform(
        labels_test
    )

    model.eval()

    with torch.no_grad():

        logits = model(X_test)

        preds = torch.argmax(
            logits,
            dim=1
        )

        y_pred = preds.cpu().numpy()

    # Accuracy
    acc = accuracy_score(
        y_true,
        y_pred
    )

    print(f"\nMLP Accuracy: {acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    print("\nMatriz de confusión:")
    print(cm_df)

def train_mlp_classifier(
    emb_train,
    labels_train,
    epochs=200,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # labels string -> int
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(labels_train)

    # numpy -> torch
    X_train = torch.tensor(
        emb_train,
        dtype=torch.float32
    ).to(device)

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    ).to(device)

    # modelo
    input_dim = emb_train.shape[1]

    model = EmbeddingClassifier(
        input_dim=input_dim,
        num_classes=len(encoder.classes_)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # training loop
    for epoch in range(epochs):

        model.train()

        logits = model(X_train)

        loss = criterion(
            logits,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f}"
            )

    return model, encoder

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim=101, num_classes=7):
        super().__init__()

        self.model = nn.Sequential(
            
            # 384 -> 512
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # salida
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    


# =========================
# EMBEDDINGS (DINOv2)
# =========================

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


def load_image(path):
    return Image.open(path).convert("RGB")


def get_embedding(image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token → embedding global
    embedding = outputs.last_hidden_state[:, 0, :]
    #embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten()


def generate_n_embeddings(df, processor, model, device, n=50):
    embeddings = []
    labels = []

    for i, (_, row) in enumerate(df.head(n).iterrows()):
        try:
            image = load_image(row["path"])
            emb = get_embedding(image, processor, model, device)

            embeddings.append(emb)
            labels.append(row["dx"])

        except Exception as e:
            print(f"Error en {row['path']}: {e}")

    return np.array(embeddings), np.array(labels)

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

def load_metadata(base_path):
    metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")
    df = pd.read_csv(metadata_path)

    print("\nDistribución original de clases:")
    print(df["dx"].value_counts())

    return df


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

def sample_per_class(df, n_per_class=50):
    df_sampled = pd.concat(
        [
            group.sample(min(len(group), n_per_class), random_state=42)
            for _, group in df.groupby("dx")
        ]
    ).reset_index(drop=True)

    print("\nDistribución tras sampling:")
    print(df_sampled["dx"].value_counts())

    return df_sampled

def compute_centroids(embeddings, labels):
    centroids = {}

    for label in np.unique(labels):
        class_embeddings = embeddings[labels == label]
        centroids[label] = class_embeddings.mean(axis=0)

    return centroids

def predict(embedding, centroids):
    best_label = None
    best_dist = float("inf")

    for label, centroid in centroids.items():
        dist = np.linalg.norm(embedding - centroid)

        if dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label

def evaluate(embeddings, labels, centroids):
    correct = 0

    for emb, true_label in zip(embeddings, labels):
        pred = predict(emb, centroids)

        if pred == true_label:
            correct += 1

    acc = correct / len(labels)
    print(f"Accuracy: {acc:.4f}")

def train_test_split_rest(df, n_train=50):
    train_list = []
    test_list = []

    for label, group in df.groupby("dx"):
        group = group.sample(frac=1, random_state=42)  # shuffle

        train = group.iloc[:n_train]
        test = group.iloc[n_train:]  # TODO el resto

        train_list.append(train)
        test_list.append(test)

    df_train = pd.concat(train_list).reset_index(drop=True)
    df_test = pd.concat(test_list).reset_index(drop=True)

    print("\nTrain distribution:")
    print(df_train["dx"].value_counts())

    print("\nTest distribution:")
    print(df_test["dx"].value_counts())

    return df_train, df_test

def confusion_matrix_report(embeddings, labels, centroids):
    y_true = []
    y_pred = []

    for emb, true_label in zip(embeddings, labels):
        pred = predict(emb, centroids)

        y_true.append(true_label)
        y_pred.append(pred)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(labels))

    cm_df = pd.DataFrame(
        cm,
        index=np.unique(labels),
        columns=np.unique(labels)
    )

    print("\nMatriz de confusión:")
    print(cm_df)

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

    df_sampled = sample_per_class(df, n_per_class=50)
    df_train, df_test = train_test_split_rest(df, n_train=50)
    emb_train, labels_train = generate_n_embeddings(
        df_train, processor, model, device, n=len(df_train)
    )

    emb_test, labels_test = generate_n_embeddings(
        df_test, processor, model, device, n=len(df_test)
    )
    analyze_pca_variance(
        emb_train
    )
    emb_train_pca, emb_test_pca, pca = apply_pca_95(
        emb_train,
        emb_test
    )

    centroids = compute_centroids(emb_train_pca, labels_train)

    evaluate(emb_test_pca, labels_test, centroids)

    confusion_matrix_report(emb_test_pca, labels_test, centroids)

    model, encoder = train_mlp_classifier(
        emb_train,
        labels_train,
        epochs=200
    )   

    evaluate_mlp(
        model,
        encoder,
        emb_test,
        labels_test
    )


if __name__ == "__main__":
    main()