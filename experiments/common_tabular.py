from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from dataset_configs.base import SplitData, TabularDataset
from dataset_configs.tabular.registry import build_tabular_dataset
from preprocess.tabular_preprocessor import PreprocessorConfig, TabularPreprocessor


@dataclass
class PreparedTabularData:
    dataset_name: str
    dataset: TabularDataset
    split: SplitData
    feature_cols: list[str]
    label_col: str
    preprocessor: TabularPreprocessor
    train_enc_df: pd.DataFrame
    test_enc_df: pd.DataFrame
    cardinalities: list[int]
    metadata: dict[str, dict[str, Any]]


def parse_key_value_overrides(items: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid override '{item}'. Expected the format KEY=VALUE."
            )
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Key cannot be empty.")
        overrides[key] = value.strip()
    return overrides


def prepare_tabular_data(
    dataset_name: str,
    random_state: int = 42,
    dataset_config_overrides: dict[str, Any] | None = None,
    preprocessor_config: PreprocessorConfig | None = None,
) -> PreparedTabularData:
    dataset = build_tabular_dataset(
        name=dataset_name,
        random_state=random_state,
        config_overrides=dataset_config_overrides,
    )

    if preprocessor_config is None:
        preprocessor_config = PreprocessorConfig(
            encode_method="ordinal_binned",
            n_bins=10,
            numeric_unique_threshold=20,
            categorical_only=True,
            use_unk=True,
        )

    df = dataset.load_raw()
    split = dataset.make_split(df)
    feature_cols = dataset.feature_cols()
    label_col = dataset.label_col()

    preprocessor = TabularPreprocessor(preprocessor_config)
    train_enc_df, test_enc_df, cardinalities, metadata = preprocessor.fit_transform(
        df_train_normal=split.df_train_normal,
        df_test=split.df_test,
        feature_cols=feature_cols,
    )

    return PreparedTabularData(
        dataset_name=dataset_name,
        dataset=dataset,
        split=split,
        feature_cols=feature_cols,
        label_col=label_col,
        preprocessor=preprocessor,
        train_enc_df=train_enc_df,
        test_enc_df=test_enc_df,
        cardinalities=cardinalities,
        metadata=metadata,
    )
