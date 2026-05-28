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
from models.mdad import ParametricMDADRec
from preprocess.tabular_preprocessor import PreprocessorConfig
from utils.metrics import evaluate_scores, save_results_csv

try:
    from experiments.common_tabular import parse_key_value_overrides, prepare_tabular_data
    from experiments.run_mdad_nonparametric_rec import (
        parse_float_tuple,
        parse_int_tuple,
        subsample_test,
    )
except ModuleNotFoundError:
    from common_tabular import parse_key_value_overrides, prepare_tabular_data
    from run_mdad_nonparametric_rec import parse_float_tuple, parse_int_tuple, subsample_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run REC-only parametric MDAD on a tabular dataset.",
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
    parser.add_argument("--mask-schedule", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--probe-times", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-probe-views", type=int, default=16)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rec-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
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


def save_runtime_csv(output_dir: Path, row: dict[str, object]) -> None:
    output_path = output_dir / "run_mdad_parametric_rec_runtime.csv"
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
    x_train = prepared.train_enc_df.to_numpy(dtype=np.int64)
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
    print(f"Train normal size: {len(x_train)}")
    print(f"Test size used: {len(x_test)}")

    model = ParametricMDADRec(
        cardinalities=prepared.cardinalities,
        mask_schedule=mask_schedule,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        eps=1e-12,
        device=device,
        seed=args.seed,
    )

    print("\nTraining parametric MDAD REC...")
    synchronize_device(device)
    train_start = time.perf_counter()
    model.fit(
        x_train,
        epochs=args.rec_epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        verbose=True,
    )
    synchronize_device(device)
    train_elapsed_seconds = time.perf_counter() - train_start

    print("\nScoring parametric MDAD REC...")
    synchronize_device(device)
    score_start = time.perf_counter()
    rec_scores = model.score_samples(
        x_test,
        probe_times=probe_times,
        n_probe_views=args.n_probe_views,
        batch_size=args.score_batch_size,
        verbose=True,
    )
    synchronize_device(device)
    score_elapsed_seconds = time.perf_counter() - score_start

    result_path = save_results_csv(
        [evaluate_scores(y_test, rec_scores, "MDAD-Parametric-REC")],
        dataset_name=args.dataset,
        filename="run_mdad_parametric_rec",
        root_dir=str(Path(args.output_root) / prepared.dataset.config.benchmark_name),
        subdirs=[str(args.seed)],
    )

if __name__ == "__main__":
    main()
