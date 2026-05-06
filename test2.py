import os
import pandas as pd
import kagglehub

from transformers import AutoImageProcessor, AutoModel

import torch
from PIL import Image
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)

from sklearn.metrics.cluster import contingency_matrix

from sklearn.mixture import GaussianMixture

from sklearn.mixture import GaussianMixture

def unsupervised_analysis_gmm(
    embeddings,
    labels
):

    print("\nNormalizando...")

    scaler = StandardScaler()

    X = scaler.fit_transform(
        embeddings
    )

    # mayor estabilidad numérica
    X = X.astype(
        np.float64
    )

    encoder = LabelEncoder()

    y_true = encoder.fit_transform(
        labels
    )

    n_classes = len(
        encoder.classes_
    )

    print(
        f"GMM con k={n_classes}"
    )

    gmm = GaussianMixture(
        n_components=n_classes,
        covariance_type="diag",
        reg_covar=1e-4,
        n_init=5,
        random_state=42
    )

    clusters = gmm.fit_predict(
        X
    )

    ari = adjusted_rand_score(
        y_true,
        clusters
    )

    nmi = normalized_mutual_info_score(
        y_true,
        clusters
    )

    print("\nRESULTADOS")

    print(
        f"ARI: {ari:.4f}"
    )

    print(
        f"NMI: {nmi:.4f}"
    )

    cm = contingency_matrix(
        y_true,
        clusters
    )

    df_cm = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=[
            f"cluster_{i}"
            for i in range(
                n_classes
            )
        ]
    )

    print(
        "\nClase real vs cluster:"
    )

    print(
        df_cm
    )

def sample_balanced(
    df,
    n_per_class=100,
    random_state=42
):
    """
    Selecciona n muestras por clase.

    Si una clase tiene menos de n,
    coge todas.
    """

    df_balanced = pd.concat(
        [
            group.sample(
                n=min(
                    len(group),
                    n_per_class
                ),
                random_state=random_state
            )
            for _, group in df.groupby(
                "dx"
            )
        ]
    ).reset_index(
        drop=True
    )

    print(
        "\nDistribución balanceada:"
    )

    print(
        df_balanced["dx"].value_counts()
    )

    print(
        f"\nTotal muestras: "
        f"{len(df_balanced)}"
    )

    return df_balanced
# =========================
# EMBEDDINGS
# =========================

def load_dinov2():

    model_name = "facebook/dinov2-base"

    processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    model.eval()

    print(f"Modelo cargado en {device}")

    return processor, model, device


def load_image(path):

    return Image.open(path).convert("RGB")


def get_embedding(
    image,
    processor,
    model,
    device
):

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model(**inputs)

    # CLS token
    #embedding = outputs.last_hidden_state[:, 0, :]
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten()


def generate_all_embeddings(
    df,
    processor,
    model,
    device
):

    embeddings = []
    labels = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 500 == 0:
            print(f"{i}/{total}")

        try:

            image = load_image(
                row["path"]
            )

            emb = get_embedding(
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
                f"Error {row['path']}: {e}"
            )

    return (
        np.array(embeddings),
        np.array(labels)
    )


# =========================
# DATASET
# =========================

def download_dataset():

    path = kagglehub.dataset_download(
        "kmader/skin-cancer-mnist-ham10000"
    )

    if "HAM10000_metadata.csv" in os.listdir(path):

        return path

    versions_path = os.path.join(
        path,
        "versions"
    )

    version = os.listdir(
        versions_path
    )[0]

    return os.path.join(
        versions_path,
        version
    )


def load_metadata(base_path):

    metadata_path = os.path.join(
        base_path,
        "HAM10000_metadata.csv"
    )

    df = pd.read_csv(
        metadata_path
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

        for img in os.listdir(folder):

            image_id = img.split(".")[0]

            image_paths[image_id] = os.path.join(
                folder,
                img
            )

    df["path"] = df["image_id"].map(
        image_paths
    )

    return df


# =========================
# UNSUPERVISED
# =========================

def unsupervised_analysis(
    embeddings,
    labels
):

    print("\nNormalizando...")

    scaler = StandardScaler()

    X = scaler.fit_transform(
        embeddings
    )

    print("PCA...")

    pca = PCA(
        n_components=0.95
    )

    X = pca.fit_transform(
        X
    )

    print(
        f"Dim PCA: {X.shape[1]}"
    )

    # labels reales -> int
    encoder = LabelEncoder()

    y_true = encoder.fit_transform(
        labels
    )

    n_classes = len(
        encoder.classes_
    )

    print(
        f"KMeans con k={n_classes}"
    )

    kmeans = KMeans(
        n_clusters=n_classes,
        random_state=42,
        n_init=20
    )

    clusters = kmeans.fit_predict(
        X
    )

    # métricas
    ari = adjusted_rand_score(
        y_true,
        clusters
    )

    nmi = normalized_mutual_info_score(
        y_true,
        clusters
    )

    print("\nRESULTADOS")
    print(
        f"ARI: {ari:.4f}"
    )

    print(
        f"NMI: {nmi:.4f}"
    )

    # tabla cluster vs clase
    cm = contingency_matrix(
        y_true,
        clusters
    )

    df_cm = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=[
            f"cluster_{i}"
            for i in range(
                n_classes
            )
        ]
    )

    print(
        "\nClase real vs cluster:"
    )

    print(
        df_cm
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

    processor, model, device = load_dinov2()
    
    df = sample_balanced(
        df,
        n_per_class=100
    )

    embeddings, labels = generate_all_embeddings(
        df,
        processor,
        model,
        device
    )
    unsupervised_analysis(
        embeddings,
        labels
    )

    unsupervised_analysis_gmm(
        embeddings,
        labels
    )



if __name__ == "__main__":
    main()