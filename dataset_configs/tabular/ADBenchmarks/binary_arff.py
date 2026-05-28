from __future__ import annotations

from typing import Protocol

import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_configs.base import SplitData, TabularDataset
from dataset_configs.tabular.arff_utils import load_arff_dataframe


class BinaryArffConfig(Protocol):
    benchmark_name: str
    arff_path: str
    raw_label_col: str
    output_label_col: str
    normal_label: str
    anomaly_label: str
    test_size: float
    feature_cols: tuple[str, ...] | None


class BinaryArffDataset(TabularDataset):
    def __init__(self, config: BinaryArffConfig, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.config = config
        self._feature_cols: list[str] | None = None

    def load_raw(self) -> pd.DataFrame:
        df = load_arff_dataframe(self.config.arff_path)

        if self.config.raw_label_col not in df.columns:
            raise ValueError(
                f"Missing label column '{self.config.raw_label_col}' in {self.config.arff_path}"
            )

        feature_cols = self._resolve_feature_cols(df)
        required = feature_cols + [self.config.raw_label_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df[required].copy()
        df = df.dropna(subset=[self.config.raw_label_col]).copy()

        allowed = {self.config.normal_label, self.config.anomaly_label}
        df = df[df[self.config.raw_label_col].isin(allowed)].copy()
        df[self.config.output_label_col] = (
            df[self.config.raw_label_col] == self.config.anomaly_label
        ).astype(int)

        return df.drop(columns=[self.config.raw_label_col]).reset_index(drop=True)

    def feature_cols(self) -> list[str]:
        if self._feature_cols is None:
            raise RuntimeError("Call load_raw() before requesting feature columns.")
        return list(self._feature_cols)

    def label_col(self) -> str:
        return self.config.output_label_col

    def make_split(self, df: pd.DataFrame) -> SplitData:
        df_train, df_test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.random_state,
            stratify=df[self.config.output_label_col],
        )
        df_train_normal = df_train[df_train[self.config.output_label_col] == 0].copy()
        return SplitData(
            df_train=df_train,
            df_train_normal=df_train_normal,
            df_val=None,
            df_test=df_test,
        )

    def _resolve_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if self.config.feature_cols is None:
            feature_cols = [col for col in df.columns if col != self.config.raw_label_col]
        else:
            feature_cols = list(self.config.feature_cols)

        self._feature_cols = feature_cols
        return feature_cols
