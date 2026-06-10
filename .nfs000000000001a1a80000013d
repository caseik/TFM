from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitDirective:
    train: int | float
    val: int | float
    test: int | float | str
    group_by: str = "lesion_id"


class SplitManager:
    """
    Gestor de particiones definitivo para HAM10000.

    Mejoras frente a las versiones previas:
    - Evita fuga de información dividiendo por `lesion_id`, no por imagen aislada.
    - Soporta few-shot por clase: train=50, val=10, test="rest".
    - Soporta porcentajes estratificados: train=0.70, val=0.15, test=0.15.
    - Mantiene el mismo test para todos los modelos si se usa el mismo seed y escenario.
    - No duplica filas para "augmentación"; la augmentación real queda en los transforms
      de PyTorch de los runners de imagen.
    - Puede exportar los CSV de train/val/test para documentar el protocolo experimental.
    """

    def __init__(
        self,
        seed: int = 42,
        split_dir: str | Path = "splits",
        export_splits: bool = True,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.split_dir = Path(split_dir)
        self.export_splits = export_splits

    def prepare_split(self, dataset: pd.DataFrame, experiment_config: dict[str, Any]) -> dict[str, pd.DataFrame]:
        directive = SplitDirective(
            train=experiment_config["train"],
            val=experiment_config.get("val", 0),
            test=experiment_config.get("test", "rest"),
            group_by=experiment_config.get("group_by", "lesion_id"),
        )

        self.validate_dataset(dataset, directive.group_by)

        if self.is_percentage_protocol(directive):
            subsets = self.select_by_percentages(dataset, directive)
        else:
            subsets = self.select_few_shot(dataset, directive)

        subsets = self.ensure_columns_and_order(subsets)

        if self.export_splits:
            self.export_split_csvs(subsets, experiment_config)

        return subsets

    def validate_dataset(self, dataset: pd.DataFrame, group_by: str) -> None:
        required = {"dx", "image_path"}
        missing = required - set(dataset.columns)
        if missing:
            raise ValueError(f"Faltan columnas obligatorias en el dataset: {sorted(missing)}")

        if group_by not in dataset.columns:
            raise ValueError(
                f"No existe la columna '{group_by}'. Para HAM10000 se recomienda dividir por lesion_id."
            )

        if dataset["image_path"].isna().any():
            raise ValueError("Existen imágenes sin ruta en image_path.")

        if dataset["dx"].isna().any():
            raise ValueError("Existen muestras sin etiqueta dx.")

    def is_percentage_protocol(self, directive: SplitDirective) -> bool:
        values = [directive.train, directive.val, directive.test]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        return all(isinstance(v, float) and 0 <= v <= 1 for v in numeric_values) and directive.test != "rest"

    def build_group_table(self, dataset: pd.DataFrame, group_by: str) -> pd.DataFrame:
        group_table = (
            dataset.groupby(group_by, as_index=False)
            .agg(dx=("dx", "first"), n_images=("image_path", "size"))
        )

        inconsistent = (
            dataset.groupby(group_by)["dx"]
            .nunique()
            .reset_index(name="n_labels")
            .query("n_labels > 1")
        )
        if len(inconsistent) > 0:
            raise ValueError(
                "Hay grupos con más de una etiqueta. Revisa lesion_id/dx antes de particionar."
            )

        return group_table

    def select_few_shot(self, dataset: pd.DataFrame, directive: SplitDirective) -> dict[str, pd.DataFrame]:
        group_table = self.build_group_table(dataset, directive.group_by)
        selected_groups: dict[str, set] = {"train": set(), "val": set(), "test": set()}

        for class_name in sorted(group_table["dx"].unique()):
            class_groups = group_table[group_table["dx"] == class_name][directive.group_by].to_numpy()
            self.rng.shuffle(class_groups)

            train_n = int(directive.train)
            val_n = int(directive.val) if directive.val != "rest" else 0

            train_groups = class_groups[:train_n]
            val_groups = class_groups[train_n:train_n + val_n]
            rest_groups = class_groups[train_n + val_n:]

            selected_groups["train"].update(train_groups)
            selected_groups["val"].update(val_groups)

            if directive.test == "rest":
                selected_groups["test"].update(rest_groups)
            else:
                test_n = int(directive.test)
                selected_groups["test"].update(rest_groups[:test_n])

        return {
            split_name: dataset[dataset[directive.group_by].isin(groups)].copy()
            for split_name, groups in selected_groups.items()
        }

    def select_by_percentages(self, dataset: pd.DataFrame, directive: SplitDirective) -> dict[str, pd.DataFrame]:
        group_table = self.build_group_table(dataset, directive.group_by)
        selected_groups: dict[str, set] = {"train": set(), "val": set(), "test": set()}

        for class_name in sorted(group_table["dx"].unique()):
            class_groups = group_table[group_table["dx"] == class_name][directive.group_by].to_numpy()
            self.rng.shuffle(class_groups)

            total = len(class_groups)
            train_n = int(round(total * float(directive.train)))
            val_n = int(round(total * float(directive.val)))

            # Asegura que no se sobrepase por redondeos.
            train_n = min(train_n, total)
            val_n = min(val_n, total - train_n)

            train_groups = class_groups[:train_n]
            val_groups = class_groups[train_n:train_n + val_n]
            test_groups = class_groups[train_n + val_n:]

            selected_groups["train"].update(train_groups)
            selected_groups["val"].update(val_groups)
            selected_groups["test"].update(test_groups)

        return {
            split_name: dataset[dataset[directive.group_by].isin(groups)].copy()
            for split_name, groups in selected_groups.items()
        }

    def ensure_columns_and_order(self, subsets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        ordered = {}
        for split_name, df in subsets.items():
            df = df.copy().reset_index(drop=True)

            if "path" not in df.columns:
                df["path"] = df["image_path"]

            # Orden estable para reproducibilidad visual y de cachés.
            sort_columns = [col for col in ["dx", "lesion_id", "image_id"] if col in df.columns]
            if sort_columns:
                df = df.sort_values(sort_columns).reset_index(drop=True)

            ordered[split_name] = df

        for required_split in ["train", "val", "test"]:
            if required_split not in ordered:
                ordered[required_split] = pd.DataFrame(columns=next(iter(ordered.values())).columns)

        return ordered

    def export_split_csvs(self, subsets: dict[str, pd.DataFrame], experiment_config: dict[str, Any]) -> None:
        scenario = experiment_config.get("scenario", "scenario")
        seed = experiment_config.get("seed", self.seed)
        split_path = self.split_dir / f"{scenario}_seed_{seed}"
        split_path.mkdir(parents=True, exist_ok=True)

        for split_name, df in subsets.items():
            export_columns = [
                col for col in [
                    "lesion_id", "image_id", "dx", "class_idx",
                    "readable_label", "image_path"
                ]
                if col in df.columns
            ]
            df[export_columns].to_csv(split_path / f"{split_name}.csv", index=False)

        summary = []
        for split_name, df in subsets.items():
            counts = df["dx"].value_counts().sort_index()
            for class_name, count in counts.items():
                summary.append({
                    "split": split_name,
                    "class": class_name,
                    "n_images": int(count),
                    "n_lesions": int(df[df["dx"] == class_name]["lesion_id"].nunique())
                    if "lesion_id" in df.columns else None,
                })

        pd.DataFrame(summary).to_csv(split_path / "split_summary.csv", index=False)
