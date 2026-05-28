from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SplitData:
    df_train: pd.DataFrame
    df_train_normal: pd.DataFrame
    df_val: Optional[pd.DataFrame]
    df_test: pd.DataFrame


class TabularDataset(ABC):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def feature_cols(self) -> list[str]:
        pass

    @abstractmethod
    def label_col(self) -> str:
        pass

    @abstractmethod
    def make_split(self, df: pd.DataFrame) -> SplitData:
        pass