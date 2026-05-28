from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import GPT2TokenizerFast

from dataset_configs.text.nlp_adbench import NLPADBenchConfig, NLPADBenchDataset
from models.mdad import ParametricMDADRecText
from utils.metrics import evaluate_scores, save_results_csv


SUPPORTED_TASKS = (
    "SMS spam classification",
    "AG News Classification",
    "To check if an email is a spam",
    "Yelp reviews dataset consists of reviews from Yelp",
)
DEFAULT_TASK = SUPPORTED_TASKS[0]
DEFAULT_MAX_LENGTH_BY_TASK = {
    "SMS spam classification": 128,
    "AG News Classification": 128,
    "To check if an email is a spam": 256,
    "Yelp reviews dataset consists of reviews from Yelp": 256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run REC-only parametric MDAD text scoring on NLP-ADBench.",
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        default=DEFAULT_TASK,
        help="Supported NLP-ADBench task to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Maximum tokenized sequence length. If omitted, uses the paper setting "
            "for the selected task: 128 for SMS/AGNews, 256 for Email/Yelp."
        ),
    )
    parser.add_argument("--mask-schedule", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--probe-times", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--n-probe-views",
        type=int,
        default=24,
        help="Number of masked reconstruction views evaluated for each probe time.",
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rec-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Limit train-normal rows for a quick smoke test; 0 uses all rows.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Limit test rows for a quick smoke test; 0 uses all rows.",
    )
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


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if len(values) < 2:
        raise ValueError("Expected at least two mask schedule values.")
    return values


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one probe time.")
    return values


def resolve_max_length(task: str, max_length: int | None) -> int:
    if max_length is not None:
        return max_length
    return DEFAULT_MAX_LENGTH_BY_TASK[task]


def subsample_train_normal(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(df):
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def subsample_test(
    df: pd.DataFrame,
    label_col: str,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(df):
        return df

    rng = np.random.default_rng(seed)
    labels = df[label_col].to_numpy(dtype=np.int64)
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        n_label = int(round(max_samples * len(indices) / len(labels)))
        n_label = max(1, min(n_label, len(indices)))
        selected.append(rng.choice(indices, size=n_label, replace=False))

    chosen = np.concatenate(selected)
    if len(chosen) > max_samples:
        chosen = rng.choice(chosen, size=max_samples, replace=False)
    elif len(chosen) < max_samples:
        remaining = np.setdiff1d(np.arange(len(df)), chosen, assume_unique=False)
        extra = rng.choice(
            remaining,
            size=min(max_samples - len(chosen), len(remaining)),
            replace=False,
        )
        chosen = np.concatenate([chosen, extra])

    rng.shuffle(chosen)
    return df.iloc[chosen].copy().reset_index(drop=True)


def tokenize_dataframe(
    tokenizer: GPT2TokenizerFast,
    df: pd.DataFrame,
    text_col: str,
    max_length: int,
) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        df[text_col].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    max_length = resolve_max_length(args.task, args.max_length)

    dataset = NLPADBenchDataset(
        config=NLPADBenchConfig(selected_task=args.task),
        random_state=args.seed,
    )
    df = dataset.load_raw()
    if df.empty:
        raise ValueError(f"No rows found for task {args.task!r}.")
    split = dataset.make_split(df)
    text_col = dataset.text_col()
    label_col = dataset.label_col()
    train_normal = subsample_train_normal(
        split.df_train_normal,
        args.max_train_samples,
        args.seed,
    )
    test_df = subsample_test(
        split.df_test,
        label_col,
        args.max_test_samples,
        args.seed,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.padding_side = "right"
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<|pad|>"
    if tokenizer.mask_token is None:
        special_tokens["mask_token"] = "<|mask|>"
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    train_tokens = tokenize_dataframe(tokenizer, train_normal, text_col, max_length)
    test_tokens = tokenize_dataframe(tokenizer, test_df, text_col, max_length)
    y_test = test_df[label_col].to_numpy(dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mask_schedule = parse_float_tuple(args.mask_schedule)
    probe_times = parse_int_tuple(args.probe_times)

    print(f"Task: {args.task}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Tokenizer: gpt2 (vocab size {len(tokenizer)})")
    print(f"Max length: {max_length}")
    print(f"Train normal size used: {len(train_normal)}")
    print(f"Test size used: {len(test_df)}")

    model = ParametricMDADRecText(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        max_length=max_length,
        mask_schedule=mask_schedule,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        device=device,
        seed=args.seed,
    )

    print("\nTraining parametric MDAD text REC...")
    model.fit_model(
        input_ids=train_tokens["input_ids"],
        attention_mask=train_tokens["attention_mask"],
        epochs=args.rec_epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        verbose=True,
    )

    print("\nScoring parametric MDAD text REC...")
    scores = model.score_samples(
        input_ids=test_tokens["input_ids"],
        attention_mask=test_tokens["attention_mask"],
        batch_size=args.score_batch_size,
        probe_times=probe_times,
        n_probe_views=args.n_probe_views,
        verbose=True,
    )
    save_results_csv(
        [evaluate_scores(y_test, scores, "MDAD-Parametric-REC-Text")],
        dataset_name="nlp_adbench",
        filename="run_mdad_parametric_rec_text",
        root_dir=args.output_root,
        subdirs=[str(args.seed), args.task],
    )


if __name__ == "__main__":
    main()
