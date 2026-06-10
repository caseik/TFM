from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def build_runner_registry(seed: int = 42) -> dict[str, Any]:
    """
    Registro único de modelos.

    Se mantiene la batería amplia de contenido_ismael y se elimina el runner genérico
    antiguo para que cada arquitectura tenga nombre propio en tablas y resultados.
    """
    return {
        "centroid": CentroidRunner(),
        "mlp": MLPRunner(),
        "efficientnet_b0": EfficientNetB0Runner(seed=seed),
        "convnext_tiny": ConvNeXtTinyRunner(seed=seed),
        "mobilenet_v2": MobileNetV2Runner(seed=seed),
        "vit_b16": ViTB16Runner(seed=seed),
        "resnet50": ResNet50Runner(seed=seed),
        "densenet121": DenseNet121Runner(seed=seed),
        "inception_v3": InceptionV3Runner(seed=seed),
    }


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
    feature_extractor = FeatureExtractor(cache_dir=results_dir / "cache")

    for experiment in experiments:
        print("\n" + "=" * 80)
        print(f"Ejecutando experimento: {experiment['experiment_id']}")
        print("=" * 80)

        seed = int(experiment["seed"])

        split_manager = SplitManager(
            seed=seed,
            split_dir=results_dir / "splits",
            export_splits=True,
        )
        split_bundle = split_manager.prepare_split(dataset, experiment)

        if experiment["input_mode"] == "embeddings":
            split_bundle = feature_extractor.transform_split_bundle(
                split_bundle,
                experiment["feature_extractor"],
            )

        runners = build_runner_registry(seed=seed)
        runner = runners[experiment["runner"]]

        predictions = runner.run(split_bundle)

        metrics_generator.process_experiment(
            experiment,
            predictions,
        )

        metrics_generator.export_tables()

    performance, false_negatives = metrics_generator.export_tables()

    print("\nTablas generadas:")
    print(results_dir / "table_2_performance.csv")
    print(results_dir / "table_3_false_negatives.csv")
    print("\nResumen de rendimiento:")
    print(performance[["Experiment ID", "Accuracy", "Balanced Accuracy", "Macro F1"]])


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
    experiment_logger.export_table(results_dir / "table_1_experimental_plan.csv")

    run_experiments(
        dataset=dataset,
        experiments=experiments,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
