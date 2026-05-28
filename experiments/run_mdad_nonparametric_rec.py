from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from dataset_configs.tabular.registry import list_tabular_datasets
from models.mdad import NonParametricMDADRec
from preprocess.tabular_preprocessor import PreprocessorConfig
from utils.metrics import evaluate_scores, save_results_csv

try:
    from experiments.common_tabular import parse_key_value_overrides, prepare_tabular_data
except ModuleNotFoundError:
    from common_tabular import parse_key_value_overrides, prepare_tabular_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run REC-only nonparametric MDAD on a tabular dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="u2r",
        choices=list_tabular_datasets(),
        help="Registered tabular dataset name.",
    )
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--dataset-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--mask-schedule", default="0.0,0.15,0.30,0.45,0.60")
    parser.add_argument("--probe-times", default="1,2,3,4")
    parser.add_argument(
        "--n-probe-views",
        type=int,
        default=8,
        help="Number of masked reconstruction views evaluated for each probe time.",
    )
    parser.add_argument("--lambda-kernel", type=float, default=1.0)
    parser.add_argument("--ref-subsample", type=int, default=500000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ref-chunk-size", type=int, default=4096)
    parser.add_argument("--output-root", default="results")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) < 2:
        raise ValueError("Expected at least two values.")
    return values


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def subsample_test(
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(x_test):
        return x_test, y_test

    rng = np.random.default_rng(seed)
    classes = np.unique(y_test)
    selected: list[np.ndarray] = []

    for cls in classes:
        cls_idx = np.flatnonzero(y_test == cls)
        n_cls = int(round(max_samples * len(cls_idx) / len(y_test)))
        n_cls = max(1, min(n_cls, len(cls_idx)))
        selected.append(rng.choice(cls_idx, size=n_cls, replace=False))

    idx = np.concatenate(selected)
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    elif len(idx) < max_samples:
        remaining = np.setdiff1d(np.arange(len(x_test)), idx, assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(
                remaining,
                size=min(max_samples - len(idx), len(remaining)),
                replace=False,
            )
            idx = np.concatenate([idx, extra])

    rng.shuffle(idx)
    return x_test[idx], y_test[idx]


def save_runtime_csv(output_dir: Path, row: dict[str, object]) -> None:
    output_path = output_dir / "run_mdad_nonparametric_rec_runtime.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved runtime to {output_path}")


def main():
    args = parse_args()
    if args.random_state is not None:
        args.seed = args.random_state
    set_global_seed(args.seed)

    prepared = prepare_tabular_data(
        dataset_name=args.dataset,
        random_state=args.seed,
        dataset_config_overrides=parse_key_value_overrides(args.dataset_config),
        preprocessor_config=PreprocessorConfig(
            encode_method="ordinal_binned",
            n_bins=10,
            numeric_unique_threshold=20,
            categorical_only=True,
            use_unk=True,
        ),
    )

    split = prepared.split
    x_train_full = prepared.train_enc_df.to_numpy(dtype=np.int64)
    x_test = prepared.test_enc_df.to_numpy(dtype=np.int64)
    y_test = split.df_test[prepared.label_col].to_numpy(dtype=np.int64)
    x_test, y_test = subsample_test(x_test, y_test, args.max_test_samples, args.seed)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mask_schedule = parse_float_tuple(args.mask_schedule)
    probe_times = parse_int_tuple(args.probe_times)

    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Train normal size: {len(x_train_full)}")
    print(f"Test size used: {len(x_test)}")

    if args.ref_subsample > 0 and args.ref_subsample < len(x_train_full):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(x_train_full), size=args.ref_subsample, replace=False)
        x_train = x_train_full[idx]
        print(f"Reference subsample: {len(x_train)} / {len(x_train_full)}")
    else:
        x_train = x_train_full
        print(f"Reference uses full train-normal set: {len(x_train)}")

    model = NonParametricMDADRec(
        cardinalities=prepared.cardinalities,
        mask_schedule=mask_schedule,
        lambda_kernel=args.lambda_kernel,
        eps=1e-12,
        device=device,
        seed=args.seed,
    )
    model.fit(x_train)

    print("\nScoring nonparametric MDAD REC...")
    synchronize_device(device)
    score_start = time.perf_counter()
    rec_scores = model.score_samples(
        x_test,
        probe_times=probe_times,
        n_probe_views=args.n_probe_views,
        batch_size=args.batch_size,
        ref_chunk_size=args.ref_chunk_size,
        verbose=True,
    )
    synchronize_device(device)
    score_elapsed_seconds = time.perf_counter() - score_start

    result_path = save_results_csv(
        [evaluate_scores(y_test, rec_scores, "MDAD-NonParametric-REC")],
        dataset_name=args.dataset,
        filename="run_mdad_nonparametric_rec",
        root_dir=str(Path(args.output_root) / prepared.dataset.config.benchmark_name),
        subdirs=[str(args.seed)],
    )

if __name__ == "__main__":
    main()
