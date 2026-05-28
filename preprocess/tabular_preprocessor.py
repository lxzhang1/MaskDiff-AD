from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PreprocessorConfig:
    """
    encode_method:
        - 'ordinal_binned': categorical -> integer ids, numeric -> quantile bins -> integer ids
        - 'onehot': categorical -> one-hot, numeric -> median impute and keep continuous
        - 'binned_onehot': categorical -> one-hot, numeric -> quantile bins -> one-hot
    """
    encode_method: str = "ordinal_binned"
    n_bins: int = 10
    numeric_unique_threshold: int = 20
    categorical_only: bool = False
    use_unk: bool = True
    unk_token: str = "__UNK__"
    dummy_na_token: str = "__NA__"


class TabularPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config

        self.feature_cols_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []

        # per-column fitted metadata
        self.metadata_: dict[str, dict[str, Any]] = {}

        # output metadata
        self.cardinalities_: list[int] = []
        self.output_feature_names_: list[str] = []

        self._is_fitted: bool = False

    # ============================================================
    # Public API
    # ============================================================
    def fit(self, df_train_normal: pd.DataFrame, feature_cols: list[str]) -> "TabularPreprocessor":
        self.feature_cols_ = list(feature_cols)
        if self.config.categorical_only:
            self.numeric_cols_ = []
            self.categorical_cols_ = list(self.feature_cols_)
        else:
            self.numeric_cols_, self.categorical_cols_ = self.detect_column_types(
                df_train_normal, self.feature_cols_
            )

        self.metadata_ = {}

        if self.config.encode_method == "ordinal_binned":
            self._fit_ordinal_binned(df_train_normal)
        elif self.config.encode_method == "onehot":
            self._fit_onehot(df_train_normal)
        elif self.config.encode_method == "binned_onehot":
            self._fit_binned_onehot(df_train_normal)
        else:
            raise ValueError(
                f"Unknown encode_method={self.config.encode_method}. "
                f"Supported: 'ordinal_binned', 'onehot', 'binned_onehot'."
            )

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()

        if self.config.encode_method == "ordinal_binned":
            return self._transform_ordinal_binned(df)
        elif self.config.encode_method == "onehot":
            return self._transform_onehot(df)
        elif self.config.encode_method == "binned_onehot":
            return self._transform_binned_onehot(df)
        else:
            raise ValueError(f"Unsupported encode_method={self.config.encode_method}")

    def fit_transform(
        self,
        df_train_normal: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_cols: list[str],
    ):
        self.fit(df_train_normal, feature_cols)
        train_enc = self.transform(df_train_normal)
        test_enc = self.transform(df_test)
        return train_enc, test_enc, self.cardinalities_, self.metadata_

    # ============================================================
    # Column typing
    # ============================================================
    def detect_column_types(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        numeric_cols = []
        categorical_cols = []

        for col in feature_cols:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                nunique = s.nunique(dropna=True)
                if nunique <= self.config.numeric_unique_threshold:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols

    # ============================================================
    # ordinal_binned
    # ============================================================
    def _fit_ordinal_binned(self, df_train_normal: pd.DataFrame) -> None:
        self.output_feature_names_ = list(self.feature_cols_)
        self.cardinalities_ = []

        # numeric -> quantile bins fitted on train-normal only
        for col in self.numeric_cols_:
            train_num = pd.to_numeric(df_train_normal[col], errors="coerce")
            median_val = train_num.median()
            train_num = train_num.fillna(median_val)

            try:
                _, bin_edges = pd.qcut(
                    train_num,
                    q=self.config.n_bins,
                    retbins=True,
                    duplicates="drop",
                )
            except ValueError:
                uniq = np.unique(train_num.to_numpy())
                if len(uniq) <= 1:
                    bin_edges = np.array([-np.inf, np.inf], dtype=float)
                else:
                    qs = np.linspace(0, 1, min(self.config.n_bins, len(uniq)) + 1)
                    bin_edges = np.unique(np.quantile(train_num.to_numpy(), qs))
                    if len(bin_edges) < 2:
                        bin_edges = np.array([-np.inf, np.inf], dtype=float)

            bin_edges = np.asarray(bin_edges, dtype=float).copy()
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            card = len(bin_edges) - 1
            self.metadata_[col] = {
                "type": "numeric_binned",
                "median": float(median_val),
                "bin_edges": bin_edges,
                "cardinality": card,
            }
            self.cardinalities_.append(card)

        # categorical -> integer ids, vocab fit on train-normal only
        for col in self.categorical_cols_:
            train_cat = self._clean_categorical_series(df_train_normal[col])

            cats = pd.Index(train_cat.unique())
            cat_to_id = {v: i for i, v in enumerate(cats)}

            cardinality = len(cats)
            unk_id = None
            if self.config.use_unk:
                unk_id = cardinality
                cardinality += 1

            self.metadata_[col] = {
                "type": "categorical",
                "categories": list(cats),
                "cat_to_id": cat_to_id,
                "unk_id": unk_id,
                "cardinality": cardinality,
            }
            self.cardinalities_.append(cardinality)

        # reorder cardinalities to match feature_cols_
        self.cardinalities_ = [
            self.metadata_[col]["cardinality"] for col in self.feature_cols_
        ]

    def _transform_ordinal_binned(self, df: pd.DataFrame) -> pd.DataFrame:
        out_data: dict[str, pd.Series] = {}

        for col in self.feature_cols_:
            meta = self.metadata_[col]
            if meta["type"] == "numeric_binned":
                x = pd.to_numeric(df[col], errors="coerce")
                x = x.fillna(meta["median"])
                codes = pd.cut(
                    x,
                    bins=meta["bin_edges"],
                    labels=False,
                    include_lowest=True,
                    right=True,
                ).astype(np.int64)
                out_data[col] = codes

            elif meta["type"] == "categorical":
                x = self._clean_categorical_series(df[col])
                mapped = x.map(meta["cat_to_id"])
                if meta["unk_id"] is not None:
                    mapped = mapped.fillna(meta["unk_id"])
                else:
                    # if no UNK is allowed, unseen values raise
                    if mapped.isna().any():
                        unseen = x[mapped.isna()].unique().tolist()
                        raise ValueError(
                            f"Unseen categories in column '{col}' during transform: {unseen[:10]}"
                        )
                out_data[col] = mapped.astype(np.int64)

            else:
                raise ValueError(f"Unknown metadata type for column {col}: {meta['type']}")

        out = pd.DataFrame(out_data, index=df.index)
        return out[self.feature_cols_]

    # ============================================================
    # onehot
    # ============================================================
    def _fit_onehot(self, df_train_normal: pd.DataFrame) -> None:
        self.output_feature_names_ = []
        self.cardinalities_ = []

        # numeric -> median impute, keep continuous
        for col in self.numeric_cols_:
            train_num = pd.to_numeric(df_train_normal[col], errors="coerce")
            median_val = train_num.median()
            self.metadata_[col] = {
                "type": "numeric_continuous",
                "median": float(median_val),
                "cardinality": 1,
            }
            self.output_feature_names_.append(col)

        # categorical -> one-hot vocab fit on train-normal only
        for col in self.categorical_cols_:
            train_cat = self._clean_categorical_series(df_train_normal[col])

            cats = pd.Index(train_cat.unique())
            categories = list(cats)

            # for onehot, if use_unk=True, we reserve one explicit UNK bucket
            if self.config.use_unk and self.config.unk_token not in categories:
                categories = categories + [self.config.unk_token]

            onehot_feature_names = [f"{col}__{v}" for v in categories]

            self.metadata_[col] = {
                "type": "categorical_onehot",
                "categories": categories,
                "category_set": set(categories),
                "cardinality": len(categories),
                "onehot_feature_names": onehot_feature_names,
            }
            self.output_feature_names_.extend(onehot_feature_names)

        # onehot output is a flat feature vector, so cardinalities are not used
        # in the same way as ordinal_binned. We keep them for consistency.
        self.cardinalities_ = [1] * len(self.output_feature_names_)

    def _transform_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []

        # numeric continuous block
        if len(self.numeric_cols_) > 0:
            num_df = pd.DataFrame(index=df.index)
            for col in self.numeric_cols_:
                meta = self.metadata_[col]
                x = pd.to_numeric(df[col], errors="coerce").fillna(meta["median"])
                num_df[col] = x.astype(np.float32)
            blocks.append(num_df)

        # categorical onehot block
        for col in self.categorical_cols_:
            meta = self.metadata_[col]
            x = self._clean_categorical_series(df[col])

            categories = meta["categories"]
            cat_set = meta["category_set"]

            if self.config.use_unk:
                x = x.where(x.isin(cat_set), other=self.config.unk_token)
            else:
                unseen_mask = ~x.isin(cat_set)
                if unseen_mask.any():
                    unseen = x[unseen_mask].unique().tolist()
                    raise ValueError(
                        f"Unseen categories in column '{col}' during transform: {unseen[:10]}"
                    )

            onehot = pd.DataFrame(
                0.0,
                index=df.index,
                columns=meta["onehot_feature_names"],
                dtype=np.float32,
            )

            for cat in categories:
                onehot[f"{col}__{cat}"] = (x == cat).astype(np.float32)

            blocks.append(onehot)

        if len(blocks) == 0:
            raise ValueError("No transformed blocks were produced.")

        out = pd.concat(blocks, axis=1)
        out = out[self.output_feature_names_]
        return out

    # ============================================================
    # binned_onehot
    # ============================================================
    def _fit_binned_onehot(self, df_train_normal: pd.DataFrame) -> None:
        self.output_feature_names_ = []
        self.cardinalities_ = []

        # numeric -> quantile bins fitted on train-normal only -> one-hot
        for col in self.numeric_cols_:
            train_num = pd.to_numeric(df_train_normal[col], errors="coerce")
            median_val = train_num.median()
            train_num = train_num.fillna(median_val)

            try:
                _, bin_edges = pd.qcut(
                    train_num,
                    q=self.config.n_bins,
                    retbins=True,
                    duplicates="drop",
                )
            except ValueError:
                uniq = np.unique(train_num.to_numpy())
                if len(uniq) <= 1:
                    bin_edges = np.array([-np.inf, np.inf], dtype=float)
                else:
                    qs = np.linspace(0, 1, min(self.config.n_bins, len(uniq)) + 1)
                    bin_edges = np.unique(np.quantile(train_num.to_numpy(), qs))
                    if len(bin_edges) < 2:
                        bin_edges = np.array([-np.inf, np.inf], dtype=float)

            bin_edges = np.asarray(bin_edges, dtype=float).copy()
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            card = len(bin_edges) - 1
            onehot_feature_names = [f"{col}__bin_{bin_id}" for bin_id in range(card)]
            self.metadata_[col] = {
                "type": "numeric_binned_onehot",
                "median": float(median_val),
                "bin_edges": bin_edges,
                "cardinality": card,
                "onehot_feature_names": onehot_feature_names,
            }
            self.output_feature_names_.extend(onehot_feature_names)

        # categorical -> one-hot vocab fit on train-normal only
        for col in self.categorical_cols_:
            train_cat = self._clean_categorical_series(df_train_normal[col])

            cats = pd.Index(train_cat.unique())
            categories = list(cats)

            if self.config.use_unk and self.config.unk_token not in categories:
                categories = categories + [self.config.unk_token]

            onehot_feature_names = [f"{col}__{v}" for v in categories]

            self.metadata_[col] = {
                "type": "categorical_onehot",
                "categories": categories,
                "category_set": set(categories),
                "cardinality": len(categories),
                "onehot_feature_names": onehot_feature_names,
            }
            self.output_feature_names_.extend(onehot_feature_names)

        self.cardinalities_ = [1] * len(self.output_feature_names_)

    def _transform_binned_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []

        for col in self.feature_cols_:
            meta = self.metadata_[col]

            if meta["type"] == "numeric_binned_onehot":
                x = pd.to_numeric(df[col], errors="coerce")
                x = x.fillna(meta["median"])
                codes = pd.cut(
                    x,
                    bins=meta["bin_edges"],
                    labels=False,
                    include_lowest=True,
                    right=True,
                ).astype(np.int64)

                onehot = pd.DataFrame(
                    0.0,
                    index=df.index,
                    columns=meta["onehot_feature_names"],
                    dtype=np.float32,
                )
                for bin_id in range(meta["cardinality"]):
                    onehot[f"{col}__bin_{bin_id}"] = (codes == bin_id).astype(np.float32)
                blocks.append(onehot)

            elif meta["type"] == "categorical_onehot":
                x = self._clean_categorical_series(df[col])

                categories = meta["categories"]
                cat_set = meta["category_set"]

                if self.config.use_unk:
                    x = x.where(x.isin(cat_set), other=self.config.unk_token)
                else:
                    unseen_mask = ~x.isin(cat_set)
                    if unseen_mask.any():
                        unseen = x[unseen_mask].unique().tolist()
                        raise ValueError(
                            f"Unseen categories in column '{col}' during transform: {unseen[:10]}"
                        )

                onehot = pd.DataFrame(
                    0.0,
                    index=df.index,
                    columns=meta["onehot_feature_names"],
                    dtype=np.float32,
                )

                for cat in categories:
                    onehot[f"{col}__{cat}"] = (x == cat).astype(np.float32)

                blocks.append(onehot)

            else:
                raise ValueError(f"Unknown metadata type for column {col}: {meta['type']}")

        if len(blocks) == 0:
            raise ValueError("No transformed blocks were produced.")

        out = pd.concat(blocks, axis=1)
        out = out[self.output_feature_names_]
        return out

    # ============================================================
    # Helpers
    # ============================================================
    def _clean_categorical_series(self, s: pd.Series) -> pd.Series:
        s = s.copy()
        s = s.replace("?", np.nan)
        s = s.fillna(self.config.dummy_na_token)
        s = s.astype(str)
        return s

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TabularPreprocessor is not fitted yet.")
