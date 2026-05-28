from __future__ import annotations

from dataclasses import dataclass

from dataset_configs.tabular.ADBenchmarks.binary_arff import BinaryArffDataset


@dataclass
class BankConfig:
    benchmark_name: str = "ADBenchmarks"
    arff_path: str = "data/tabular/ADBenchmarks/bank-additional-ful-nominal.arff"
    raw_label_col: str = "y"
    output_label_col: str = "Label"
    normal_label: str = "no"
    anomaly_label: str = "yes"
    test_size: float = 0.30
    feature_cols: tuple[str, ...] | None = None


class BankDataset(BinaryArffDataset):
    pass
