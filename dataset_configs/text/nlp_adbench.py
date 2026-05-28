from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset

from dataset_configs.base import SplitData, TabularDataset


@dataclass
class NLPADBenchConfig:
    dataset_name: str = "kendx/NLP-ADBench"
    dataset_config_name: str = "default"
    text_col: str = "text"
    label_col: str = "label"
    task_col: str = "original_task"
    original_label_col: str = "original_label"
    selected_task: str | None = None


class NLPADBenchDataset(TabularDataset):
    def __init__(self, config: NLPADBenchConfig, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.config = config

    def load_raw(self) -> pd.DataFrame:
        dataset = load_dataset(self.config.dataset_name, self.config.dataset_config_name)
        train_df = dataset["train"].to_pandas().copy()
        test_df = dataset["test"].to_pandas().copy()
        train_df["_split"] = "train"
        test_df["_split"] = "test"
        df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        required = [
            self.config.text_col,
            self.config.label_col,
            self.config.task_col,
            self.config.original_label_col,
            "_split",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[required].copy()
        df = df.dropna(
            subset=[self.config.text_col, self.config.label_col, self.config.task_col]
        ).copy()
        df[self.config.text_col] = df[self.config.text_col].astype(str)
        df[self.config.task_col] = df[self.config.task_col].astype(str)
        df[self.config.original_label_col] = df[self.config.original_label_col].astype(str)
        df[self.config.label_col] = pd.to_numeric(df[self.config.label_col], errors="coerce")
        df = df.dropna(subset=[self.config.label_col]).copy()
        df[self.config.label_col] = df[self.config.label_col].astype(int)
        df = df[df[self.config.label_col].isin([0, 1])].copy()
        df["Label"] = df[self.config.label_col]

        if self.config.selected_task is not None:
            df = df[df[self.config.task_col] == self.config.selected_task].copy()

        return df.reset_index(drop=True)

    def feature_cols(self) -> list[str]:
        return [self.config.text_col]

    def label_col(self) -> str:
        return "Label"

    def text_col(self) -> str:
        return self.config.text_col

    def task_col(self) -> str:
        return self.config.task_col

    def make_split(self, df: pd.DataFrame) -> SplitData:
        df_train = df[df["_split"] == "train"].copy().reset_index(drop=True)
        df_test = df[df["_split"] == "test"].copy().reset_index(drop=True)
        df_train_normal = df_train[df_train["Label"] == 0].copy().reset_index(drop=True)

        return SplitData(
            df_train=df_train,
            df_train_normal=df_train_normal,
            df_val=None,
            df_test=df_test,
        )
