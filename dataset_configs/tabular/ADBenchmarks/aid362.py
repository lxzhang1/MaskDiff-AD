from __future__ import annotations

from dataclasses import dataclass

from dataset_configs.tabular.ADBenchmarks.binary_arff import BinaryArffDataset


@dataclass
class AID362Config:
    benchmark_name: str = "ADBenchmarks"
    arff_path: str = "data/tabular/ADBenchmarks/AID362red_train_allpossiblenominal.arff"
    raw_label_col: str = "Outcome"
    output_label_col: str = "Label"
    normal_label: str = "Inactive"
    anomaly_label: str = "Active"
    test_size: float = 0.30
    feature_cols: tuple[str, ...] | None = None


class AID362Dataset(BinaryArffDataset):
    pass
