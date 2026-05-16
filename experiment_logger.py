from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ExperimentRecord:
    input_mode: str
    runner: str
    feature_extractor: str
    train: int
    val: int
    test: int
    lora: bool
    augmentation: bool


class ExperimentLogger:
    def __init__(self):
        self._records = []

    def register_batch(self, experiments):
        for experiment in experiments:
            record = ExperimentRecord(
                input_mode=experiment["input_mode"],
                runner=experiment["runner"],
                feature_extractor=experiment["feature_extractor"],
                train=experiment["train"],
                val=experiment["val"],
                test=experiment["test"],
                lora=experiment["lora"],
                augmentation=experiment["augmentation"]
            )
            self._records.append(record)

    def format_records(self):
        return [asdict(record) for record in self._records]

    def export_table(self, path):
        table = pd.DataFrame(self.format_records())
        table.to_csv(path, index=False)
        return table