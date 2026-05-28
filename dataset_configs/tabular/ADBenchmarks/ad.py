from __future__ import annotations

from dataclasses import dataclass

from dataset_configs.tabular.ADBenchmarks.binary_arff import BinaryArffDataset


@dataclass
class ADConfig:
    benchmark_name: str = "ADBenchmarks"
    arff_path: str = "data/tabular/ADBenchmarks/ad_nominal.arff"
    raw_label_col: str = "class"
    output_label_col: str = "Label"
    normal_label: str = "nonad."
    anomaly_label: str = "ad."
    test_size: float = 0.30
    feature_cols: tuple[str, ...] | None = None


class ADDataset(BinaryArffDataset):
    pass
