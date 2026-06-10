from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


class MetricsGenerator:
    """
    Métricas orientadas a una comparativa médica en HAM10000.

    Incluye:
    - accuracy global,
    - balanced accuracy,
    - macro F1,
    - weighted F1,
    - sensibilidad/recall por clase,
    - precisión por clase,
    - F1 por clase,
    - falsos negativos por clase,
    - intervalos de confianza Wilson para proporciones,
    - matriz de confusión exportable.

    La métrica clínica prioritaria para el paper debe ser la sensibilidad de melanoma
    y la tasa de falsos negativos de melanoma, no solo la accuracy global.
    """

    def __init__(
        self,
        class_order: list[str] | None = None,
        output_dir: str | Path = "results",
    ):
        self.class_order = class_order or ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        self.output_dir = Path(output_dir)
        self.performance_rows: list[dict[str, Any]] = []
        self.false_negative_rows: list[dict[str, Any]] = []

    def process_experiment(self, experiment_config: dict[str, Any], predictions: dict[str, list[int]]) -> None:
        self.validate_predictions(predictions)

        y_true = np.asarray(predictions["y_true"], dtype=int)
        y_pred = np.asarray(predictions["y_pred"], dtype=int)

        labels = list(range(len(self.class_order)))

        global_metrics = self.compute_global_metrics(y_true, y_pred, labels)
        per_class_metrics = self.compute_per_class_metrics(y_true, y_pred, labels)

        self.build_result_tables(experiment_config, global_metrics, per_class_metrics)
        self.export_confusion_matrix(experiment_config, y_true, y_pred, labels)

    def validate_predictions(self, predictions: dict[str, list[int]]) -> None:
        required_keys = {"y_true", "y_pred"}
        if not required_keys.issubset(predictions):
            raise ValueError("Invalid prediction structure. Expected y_true and y_pred.")

        if len(predictions["y_true"]) != len(predictions["y_pred"]):
            raise ValueError("Prediction size mismatch.")

        if len(predictions["y_true"]) == 0:
            raise ValueError("Empty prediction arrays.")

    def compute_global_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, Any]:
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "accuracy_ci": self.wilson_ci(int((y_true == y_pred).sum()), len(y_true)),
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "n": len(y_true),
        }

    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[int],
    ) -> dict[str, dict[str, Any]]:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        metrics: dict[str, dict[str, Any]] = {}

        for idx, class_name in enumerate(self.class_order):
            class_id = labels[idx]
            class_support = int(support[idx])
            tp = int(((y_true == class_id) & (y_pred == class_id)).sum())
            fn = int(((y_true == class_id) & (y_pred != class_id)).sum())

            metrics[class_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": class_support,
                "tp": tp,
                "fn": fn,
                "fn_rate": float(fn / class_support) if class_support else 0.0,
                "recall_ci": self.wilson_ci(tp, class_support),
                "fn_rate_ci": self.wilson_ci(fn, class_support),
            }

        return metrics

    def wilson_ci(self, successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)

        z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
        p = successes / n
        denominator = 1 + z**2 / n
        centre = p + z**2 / (2 * n)
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

        lower = (centre - margin) / denominator
        upper = (centre + margin) / denominator

        return (max(0.0, float(lower)), min(1.0, float(upper)))

    def build_result_tables(
        self,
        experiment_config: dict[str, Any],
        global_metrics: dict[str, Any],
        per_class_metrics: dict[str, dict[str, Any]],
    ) -> None:
        base = self.base_experiment_fields(experiment_config)

        performance_row = {
            **base,
            "Accuracy": self.format_percentage(global_metrics["accuracy"]),
            "Accuracy 95% CI": self.format_ci(global_metrics["accuracy_ci"]),
            "Balanced Accuracy": self.format_percentage(global_metrics["balanced_accuracy"]),
            "Macro F1": self.format_percentage(global_metrics["macro_f1"]),
            "Weighted F1": self.format_percentage(global_metrics["weighted_f1"]),
            "N test": global_metrics["n"],
        }

        fn_row = {
            **base,
            "N test": global_metrics["n"],
        }

        for class_name, class_metrics in per_class_metrics.items():
            performance_row[f"{class_name} Recall"] = self.format_percentage(class_metrics["recall"])
            performance_row[f"{class_name} Recall 95% CI"] = self.format_ci(class_metrics["recall_ci"])
            performance_row[f"{class_name} Precision"] = self.format_percentage(class_metrics["precision"])
            performance_row[f"{class_name} F1"] = self.format_percentage(class_metrics["f1"])
            performance_row[f"{class_name} n"] = class_metrics["support"]

            fn_row[f"{class_name} FN"] = class_metrics["fn"]
            fn_row[f"{class_name} FN Rate"] = self.format_percentage(class_metrics["fn_rate"])
            fn_row[f"{class_name} FN Rate 95% CI"] = self.format_ci(class_metrics["fn_rate_ci"])
            fn_row[f"{class_name} n"] = class_metrics["support"]

        self.performance_rows.append(performance_row)
        self.false_negative_rows.append(fn_row)

    def base_experiment_fields(self, experiment_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "Experiment ID": experiment_config.get("experiment_id"),
            "Scenario": experiment_config.get("scenario"),
            "Seed": experiment_config.get("seed"),
            "Input mode": experiment_config.get("input_mode"),
            "Feature extractor": experiment_config.get("feature_extractor", "none"),
            "Model": experiment_config.get("runner"),
        }

    def export_confusion_matrix(
        self,
        experiment_config: dict[str, Any],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[int],
    ) -> None:
        experiment_id = experiment_config.get("experiment_id", "experiment")
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        matrix_df = pd.DataFrame(
            matrix,
            index=[f"true_{c}" for c in self.class_order],
            columns=[f"pred_{c}" for c in self.class_order],
        )

        matrix_dir = self.output_dir / "confusion_matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        matrix_df.to_csv(matrix_dir / f"{experiment_id}.csv")

    def export_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        performance = pd.DataFrame(self.performance_rows)
        false_negatives = pd.DataFrame(self.false_negative_rows)

        performance.to_csv(self.output_dir / "table_2_performance.csv", index=False)
        false_negatives.to_csv(self.output_dir / "table_3_false_negatives.csv", index=False)

        return performance, false_negatives

    def format_percentage(self, value: float) -> str:
        return f"{value * 100:.1f}%"

    def format_ci(self, ci: tuple[float, float]) -> str:
        lower, upper = ci
        return f"{lower * 100:.1f}–{upper * 100:.1f}"
