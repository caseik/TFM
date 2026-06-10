from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from config import (
    CLASS_ORDER,
    DEFAULT_SCENARIO,
    EMBEDDING_EXPERIMENTS,
    IMAGE_EXPERIMENTS,
    SCENARIOS,
    SEEDS,
)
from dataset_manager import DatasetManager
from experiment_logger import ExperimentLogger
from feature_extractor import FeatureExtractor
from metrics_generator import MetricsGenerator
from split_manager import SplitManager

from model_runners.centroid_runner import CentroidRunner
from model_runners.convnext_tiny_runner import ConvNeXtTinyRunner
from model_runners.densenet121_runner import DenseNet121Runner
from model_runners.efficientnet_b0_runner import EfficientNetB0Runner
from model_runners.inception_v3_runner import InceptionV3Runner
from model_runners.mlp_runner import MLPRunner
from model_runners.mobilenet_v2_runner import MobileNetV2Runner
from model_runners.resnet50_runner import ResNet50Runner
from model_runners.vit_b16_runner import ViTB16Runner


def build_runner(runner_name: str, seed: int = 42) -> Any:
    runners = {
        "centroid": lambda: CentroidRunner(),
        "mlp": lambda: MLPRunner(),
        "efficientnet_b0": lambda: EfficientNetB0Runner(seed=seed),
        "convnext_tiny": lambda: ConvNeXtTinyRunner(seed=seed),
        "mobilenet_v2": lambda: MobileNetV2Runner(seed=seed),
        "vit_b16": lambda: ViTB16Runner(seed=seed),
        "resnet50": lambda: ResNet50Runner(seed=seed),
        "densenet121": lambda: DenseNet121Runner(seed=seed),
        "inception_v3": lambda: InceptionV3Runner(seed=seed),
    }

    if runner_name not in runners:
        raise ValueError(f"Runner desconocido: {runner_name}")

    return runners[runner_name]()


def build_experiment_list(
    scenario_name: str,
    seeds: list[int],
    only_embeddings: bool = False,
    only_images: bool = False,
) -> list[dict[str, Any]]:
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Escenario desconocido '{scenario_name}'. Disponibles: {list(SCENARIOS)}"
        )

    scenario = SCENARIOS[scenario_name]

    base_experiments: list[dict[str, Any]] = []

    if not only_images:
        base_experiments.extend(EMBEDDING_EXPERIMENTS)

    if not only_embeddings:
        base_experiments.extend(IMAGE_EXPERIMENTS)

    experiments = []

    for seed in seeds:
        for base in base_experiments:
            experiment = {
                **base,
                "scenario": scenario_name,
                "scenario_description": scenario["description"],
                "seed": seed,
                "train": scenario["train"],
                "val": scenario["val"],
                "test": scenario["test"],
                "group_by": scenario["group_by"],
            }

            experiment["experiment_id"] = (
                f"{scenario_name}_seed_{seed}_"
                f"{experiment['feature_extractor']}_{experiment['runner']}"
            )

            experiments.append(experiment)

    return experiments


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_experiments(
    dataset,
    experiments: list[dict[str, Any]],
    results_dir: str | Path = "results",
) -> None:
    results_dir = Path(results_dir)

    metrics_generator = MetricsGenerator(
        class_order=CLASS_ORDER,
        output_dir=results_dir,
    )

    feature_extractor = FeatureExtractor(
        cache_dir=results_dir / "cache"
    )

    total_experiments = len(experiments)

    for index, experiment in enumerate(experiments, start=1):
        print("\n" + "=" * 80)
        print(f"Experimento {index}/{total_experiments}")
        print(f"ID: {experiment['experiment_id']}")
        print(f"Modelo: {experiment['runner']}")
        print(f"Seed: {experiment['seed']}")
        print(f"Escenario: {experiment['scenario']}")
        print("=" * 80)

        seed = int(experiment["seed"])

        split_manager = SplitManager(
            seed=seed,
            split_dir=results_dir / "splits",
            export_splits=True,
        )

        split_bundle = split_manager.prepare_split(
            dataset,
            experiment,
        )

        if experiment["input_mode"] == "embeddings":
            split_bundle = feature_extractor.transform_split_bundle(
                split_bundle,
                experiment["feature_extractor"],
            )

        runner = build_runner(
            experiment["runner"],
            seed=seed,
        )

        try:
            predictions = runner.run(split_bundle)

            metrics_generator.process_experiment(
                experiment,
                predictions,
            )

            metrics_generator.export_tables()

            print(f"\nExperimento completado: {experiment['experiment_id']}")
            print(f"Resultados guardados en: {results_dir}")

        except Exception as error:
            print("\nERROR EN EXPERIMENTO")
            print(f"ID: {experiment['experiment_id']}")
            print(f"Modelo: {experiment['runner']}")
            print(f"Error: {error}")

            error_log = results_dir / "errors.log"
            error_log.parent.mkdir(parents=True, exist_ok=True)

            with open(error_log, "a", encoding="utf-8") as file:
                file.write(f"{experiment['experiment_id']} | {repr(error)}\n")

        finally:
            try:
                del runner
            except UnboundLocalError:
                pass

            try:
                del predictions
            except UnboundLocalError:
                pass

            clear_memory()

    performance, false_negatives = metrics_generator.export_tables()

    print("\n" + "=" * 80)
    print("EJECUCIÓN FINALIZADA")
    print("=" * 80)
    print(f"Tabla de rendimiento: {results_dir / 'table_2_performance.csv'}")
    print(f"Tabla de falsos negativos: {results_dir / 'table_3_false_negatives.csv'}")
    print(f"Matrices de confusión: {results_dir / 'confusion_matrices'}")

    if not performance.empty:
        print("\nResumen de rendimiento:")
        available_columns = [
            column
            for column in [
                "Experiment ID",
                "Accuracy",
                "Balanced Accuracy",
                "Macro F1",
                "Weighted F1",
                "mel Recall",
                "mel Precision",
            ]
            if column in performance.columns
        ]

        print(performance[available_columns])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparativa definitiva HAM10000: fine-tuning frente a embeddings."
    )

    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        choices=list(SCENARIOS.keys()),
        help="Escenario experimental.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Semillas para repetir experimentos.",
    )

    parser.add_argument(
        "--only-embeddings",
        action="store_true",
        help="Ejecuta solo DINOv2 + centroides/MLP.",
    )

    parser.add_argument(
        "--only-images",
        action="store_true",
        help="Ejecuta solo modelos de imagen con fine-tuning.",
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directorio de salida para tablas, splits, caché y matrices.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.only_embeddings and args.only_images:
        raise ValueError(
            "No puedes usar --only-embeddings y --only-images a la vez."
        )

    dataset_manager = DatasetManager()
    experiment_logger = ExperimentLogger()

    dataset = dataset_manager.load_dataset()

    experiments = build_experiment_list(
        scenario_name=args.scenario,
        seeds=args.seeds,
        only_embeddings=args.only_embeddings,
        only_images=args.only_images,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiment_logger.register_batch(experiments)
    experiment_logger.export_table(
        results_dir / "table_1_experimental_plan.csv"
    )

    print("\nPlan experimental generado:")
    print(f"Número de experimentos: {len(experiments)}")
    print(f"Tabla: {results_dir / 'table_1_experimental_plan.csv'}")

    run_experiments(
        dataset=dataset,
        experiments=experiments,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
