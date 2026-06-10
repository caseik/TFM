from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ExperimentRecord:
    experiment_id: str
    scenario: str
    seed: int
    input_mode: str
    runner: str
    feature_extractor: str
    train: int | float
    val: int | float
    test: int | float | str
    group_by: str
    augmentation: bool


class ExperimentLogger:
    """
    Genera la tabla de planificación experimental del TFM.

    A diferencia de la versión inicial, no depende de campos obsoletos como `lora`
    y registra todos los modelos definidos en main.py.
    """

    def __init__(self):
        self._records: list[ExperimentRecord] = []

    def register_batch(self, experiments: list[dict[str, Any]]) -> None:
        for experiment in experiments:
            record = ExperimentRecord(
                experiment_id=experiment["experiment_id"],
                scenario=experiment["scenario"],
                seed=int(experiment["seed"]),
                input_mode=experiment["input_mode"],
                runner=experiment["runner"],
                feature_extractor=str(experiment.get("feature_extractor", "none")),
                train=experiment["train"],
                val=experiment["val"],
                test=experiment["test"],
                group_by=experiment.get("group_by", "lesion_id"),
                augmentation=bool(experiment.get("augmentation", False)),
            )
            self._records.append(record)

    def format_records(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self._records]

    def export_table(self, path: str | Path) -> pd.DataFrame:
        table = pd.DataFrame(self.format_records())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        return table
