from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from typing import Any

from dataset_configs.base import TabularDataset
from dataset_configs.tabular.ADBenchmarks.ad import ADConfig, ADDataset
from dataset_configs.tabular.ADBenchmarks.aid362 import AID362Config, AID362Dataset
from dataset_configs.tabular.ADBenchmarks.apas import APASConfig, APASDataset
from dataset_configs.tabular.ADBenchmarks.bank import BankConfig, BankDataset
from dataset_configs.tabular.ADBenchmarks.celeba import CelebAConfig, CelebADataset
from dataset_configs.tabular.ADBenchmarks.chess import ChessConfig, ChessDataset
from dataset_configs.tabular.ADBenchmarks.cmc import CMCConfig, CMCDataset
from dataset_configs.tabular.ADBenchmarks.covertype import CovertypeConfig, CovertypeDataset
from dataset_configs.tabular.ADBenchmarks.census import CensusConfig, CensusDataset
from dataset_configs.tabular.ADBenchmarks.probe import ProbeConfig, ProbeDataset
from dataset_configs.tabular.ADBenchmarks.r10 import R10Config, R10Dataset
from dataset_configs.tabular.ADBenchmarks.solar import SolarConfig, SolarDataset
from dataset_configs.tabular.ADBenchmarks.u2r import U2RConfig, U2RDataset
from dataset_configs.tabular.ADBenchmarks.w7a import W7AConfig, W7ADataset


@dataclass(frozen=True)
class TabularDatasetSpec:
    dataset_cls: type[TabularDataset]
    config_cls: type


TABULAR_DATASET_REGISTRY: dict[str, TabularDatasetSpec] = {
    "ad": TabularDatasetSpec(
        dataset_cls=ADDataset,
        config_cls=ADConfig,
    ),
    "aid362": TabularDatasetSpec(
        dataset_cls=AID362Dataset,
        config_cls=AID362Config,
    ),
    "apas": TabularDatasetSpec(
        dataset_cls=APASDataset,
        config_cls=APASConfig,
    ),
    "bank": TabularDatasetSpec(
        dataset_cls=BankDataset,
        config_cls=BankConfig,
    ),
    "celeba": TabularDatasetSpec(
        dataset_cls=CelebADataset,
        config_cls=CelebAConfig,
    ),
    "chess": TabularDatasetSpec(
        dataset_cls=ChessDataset,
        config_cls=ChessConfig,
    ),
    "cmc": TabularDatasetSpec(
        dataset_cls=CMCDataset,
        config_cls=CMCConfig,
    ),
    "covertype": TabularDatasetSpec(
        dataset_cls=CovertypeDataset,
        config_cls=CovertypeConfig,
    ),
    "census": TabularDatasetSpec(
        dataset_cls=CensusDataset,
        config_cls=CensusConfig,
    ),
    "probe": TabularDatasetSpec(
        dataset_cls=ProbeDataset,
        config_cls=ProbeConfig,
    ),
    "r10": TabularDatasetSpec(
        dataset_cls=R10Dataset,
        config_cls=R10Config,
    ),
    "solar": TabularDatasetSpec(
        dataset_cls=SolarDataset,
        config_cls=SolarConfig,
    ),
    "u2r": TabularDatasetSpec(
        dataset_cls=U2RDataset,
        config_cls=U2RConfig,
    ),
    "w7a": TabularDatasetSpec(
        dataset_cls=W7ADataset,
        config_cls=W7AConfig,
    ),
}


def list_tabular_datasets() -> list[str]:
    return sorted(TABULAR_DATASET_REGISTRY)


def build_tabular_dataset(
    name: str,
    random_state: int = 42,
    config_overrides: dict[str, Any] | None = None,
) -> TabularDataset:
    if name not in TABULAR_DATASET_REGISTRY:
        available = ", ".join(list_tabular_datasets())
        raise ValueError(f"Unknown tabular dataset '{name}'. Available: {available}")

    spec = TABULAR_DATASET_REGISTRY[name]
    config_kwargs = _normalize_config_overrides(spec.config_cls, config_overrides)
    config = spec.config_cls(**config_kwargs)
    return spec.dataset_cls(config=config, random_state=random_state)


def _normalize_config_overrides(
    config_cls: type,
    config_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if not config_overrides:
        return {}

    valid_fields = {field.name for field in fields(config_cls)}
    unknown = sorted(set(config_overrides) - valid_fields)
    if unknown:
        available = ", ".join(sorted(valid_fields))
        raise ValueError(
            f"Unknown config override(s): {unknown}. "
            f"Valid fields for {config_cls.__name__}: {available}"
        )

    return {
        key: _coerce_override_value(value)
        for key, value in config_overrides.items()
    }


def _coerce_override_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
