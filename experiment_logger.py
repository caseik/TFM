from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ExperimentRecord:
    model: str
    variant: str
    train: int
    val: int
    test: int
    lora: bool
    augmentation: str


class ExperimentLogger:
    def __init__(self):
        self._records = []

    def register_batch(self, experiments):
        for experiment in experiments:
            record = ExperimentRecord(
                model=experiment["model"],
                variant=experiment["variant"],
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