from __future__ import annotations

from dataclasses import dataclass

from dataset_configs.tabular.ADBenchmarks.binary_arff import BinaryArffDataset


@dataclass
class CovertypeConfig:
    benchmark_name: str = "ADBenchmarks"
    arff_path: str = "data/tabular/ADBenchmarks/covertype_nominal_4vs123567.arff"
    raw_label_col: str = "class"
    output_label_col: str = "Label"
    normal_label: str = "0"
    anomaly_label: str = "1"
    test_size: float = 0.30
    feature_cols: tuple[str, ...] | None = None


class CovertypeDataset(BinaryArffDataset):
    pass
