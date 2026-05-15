from pathlib import Path
import pandas as pd
import kagglehub


class DatasetManager:

    LABELS = {
        "akiec": "Actinic keratoses",
        "bcc": "Basal cell carcinoma",
        "bkl": "Benign keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Nevus",
        "vasc": "Vascular lesion"
    }

    def __init__(self):
        self.dataset_dir = None

    def load_dataset(self):

        self.ensure_dataset_available()

        df = self.load_metadata()

        df = self.resolve_image_paths(df)

        self.validate_dataset(df)

        df = self.enrich_metadata(df)

        return df

    def ensure_dataset_available(self):
        """
        Descarga el dataset (si no existe) y devuelve el path local.
        KaggleHub gestiona automáticamente el cache.
        """

        self.dataset_dir = Path(
            kagglehub.dataset_download(
                "kmader/skin-cancer-mnist-ham10000"
            )
        )

    def load_metadata(self):

        metadata_path = (
            self.dataset_dir / "HAM10000_metadata.csv"
        )

        return pd.read_csv(metadata_path)

    def resolve_image_paths(self, df):

        image_paths = {}

        for image_file in self.dataset_dir.rglob("*.jpg"):

            image_paths[
                image_file.stem
            ] = str(image_file)

        df["image_path"] = df["image_id"].map(
            image_paths
        )

        return df

    def validate_dataset(self, df):

        if df["image_path"].isna().any():
            raise ValueError("Faltan imágenes.")

        if df["image_id"].duplicated().any():
            raise ValueError("Hay image_id duplicados.")

    def enrich_metadata(self, df):

        # Labels legibles
        df["readable_label"] = df["dx"].map(
            self.LABELS
        )

        # Índices numéricos
        classes = sorted(df["dx"].unique())

        class_to_idx = {
            label: idx
            for idx, label in enumerate(classes)
        }

        df["class_idx"] = df["dx"].map(
            class_to_idx
        )

        # Seleccionar solo columnas útiles
        df = df[
            [
                "lesion_id",
                "image_id",
                "dx",
                "image_path",
                "readable_label",
                "class_idx"
            ]
        ]

        return df