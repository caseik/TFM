from collections import defaultdict
from math import sqrt


class MetricsGenerator:

    def __init__(self):
        self.performance_rows = []
        self.false_negative_rows = []

    def process_experiment(
        self,
        experiment_config,
        predictions
    ):
        self.validate_predictions(
            predictions
        )

        statistics = self.build_confusion_statistics(
            predictions
        )

        metrics = self.compute_metrics(
            statistics
        )

        metrics = self.estimate_confidence_intervals(
            metrics
        )

        self.build_result_tables(
            experiment_config,
            metrics
        )

    def validate_predictions(
        self,
        predictions
    ):
        required_keys = {
            "y_true",
            "y_pred"
        }

        if not required_keys.issubset(
            predictions
        ):
            raise ValueError(
                "Invalid prediction structure"
            )

        if len(predictions["y_true"]) != len(
            predictions["y_pred"]
        ):
            raise ValueError(
                "Prediction size mismatch"
            )

    def build_confusion_statistics(
        self,
        predictions
    ):
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]

        classes = sorted(
            set(y_true)
        )

        statistics = {
            "classes": classes,
            "global": {
                "correct": 0,
                "total": len(y_true)
            },
            "per_class": defaultdict(
                lambda: {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "support": 0
                }
            )
        }

        for true_label, pred_label in zip(
            y_true,
            y_pred
        ):
            statistics["per_class"][
                true_label
            ]["support"] += 1

            if true_label == pred_label:
                statistics["global"][
                    "correct"
                ] += 1

                statistics["per_class"][
                    true_label
                ]["tp"] += 1
            else:
                statistics["per_class"][
                    true_label
                ]["fn"] += 1

                statistics["per_class"][
                    pred_label
                ]["fp"] += 1

        return statistics

    def compute_metrics(
        self,
        statistics
    ):
        metrics = {
            "accuracy": {},
            "per_class": {}
        }

        correct = statistics["global"][
            "correct"
        ]

        total = statistics["global"][
            "total"
        ]

        metrics["accuracy"] = {
            "value": correct / total,
            "n": total
        }

        for class_name in statistics[
            "classes"
        ]:
            class_stats = statistics[
                "per_class"
            ][class_name]

            tp = class_stats["tp"]
            fn = class_stats["fn"]
            support = class_stats[
                "support"
            ]

            sensitivity = (
                tp / support
                if support > 0
                else 0
            )

            fn_rate = (
                fn / support
                if support > 0
                else 0
            )

            metrics["per_class"][
                class_name
            ] = {
                "sensitivity": sensitivity,
                "fn_rate": fn_rate,
                "fn": fn,
                "support": support
            }

        return metrics

    def estimate_confidence_intervals(
        self,
        metrics
    ):
        metrics["accuracy"][
            "ci"
        ] = self.compute_ci(
            metrics["accuracy"]["value"],
            metrics["accuracy"]["n"]
        )

        for class_name in metrics[
            "per_class"
        ]:
            class_metrics = metrics[
                "per_class"
            ][class_name]

            support = class_metrics[
                "support"
            ]

            class_metrics[
                "sensitivity_ci"
            ] = self.compute_ci(
                class_metrics[
                    "sensitivity"
                ],
                support
            )

            class_metrics[
                "fn_rate_ci"
            ] = self.compute_ci(
                class_metrics[
                    "fn_rate"
                ],
                support
            )

        return metrics

    def compute_ci(
        self,
        proportion,
        n,
        z=1.96
    ):
        if n == 0:
            return (0, 0)

        margin = z * sqrt(
            (
                proportion
                * (1 - proportion)
            ) / n
        )

        lower = max(
            0,
            proportion - margin
        )

        upper = min(
            1,
            proportion + margin
        )

        return (
            lower,
            upper
        )

    def build_result_tables(
        self,
        experiment_config,
        metrics
    ):
        model_name = experiment_config[
            "runner"
        ]

        performance_row = {
            "Model": model_name,
            "Global Acc.": self.format_percentage(
                metrics["accuracy"]["value"]
            ),
            "95% CI": self.format_ci(
                metrics["accuracy"]["ci"]
            )
        }

        fn_row = {
            "Model": model_name
        }

        for class_name, class_metrics in metrics[
            "per_class"
        ].items():

            performance_row[
                f"{class_name} Sens."
            ] = self.format_percentage(
                class_metrics[
                    "sensitivity"
                ]
            )

            performance_row[
                f"{class_name} CI"
            ] = self.format_ci(
                class_metrics[
                    "sensitivity_ci"
                ]
            )

            performance_row[
                f"{class_name} n"
            ] = class_metrics[
                "support"
            ]

            fn_row[
                f"{class_name} FN"
            ] = class_metrics[
                "fn"
            ]

            fn_row[
                f"{class_name} FN Rate"
            ] = self.format_percentage(
                class_metrics[
                    "fn_rate"
                ]
            )

            fn_row[
                f"{class_name} CI"
            ] = self.format_ci(
                class_metrics[
                    "fn_rate_ci"
                ]
            )

            fn_row[
                f"{class_name} n"
            ] = class_metrics[
                "support"
            ]

        self.performance_rows.append(
            performance_row
        )

        self.false_negative_rows.append(
            fn_row
        )

    def format_percentage(
        self,
        value
    ):
        return f"{value * 100:.1f}%"

    def format_ci(
        self,
        ci
    ):
        lower, upper = ci

        return (
            f"{lower * 100:.1f}"
            f"–"
            f"{upper * 100:.1f}"
        )