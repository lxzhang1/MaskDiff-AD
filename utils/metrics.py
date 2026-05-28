from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_score_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    
    return {
        "ROC-AUC": float(auc),
        "PR-AUC": float(ap),
    }


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, name: str) -> dict[str, float | str]:
    metrics = compute_score_metrics(y_true, scores)
    print(
        f"{name:>16s} | ROC-AUC: {metrics['ROC-AUC']:.4f} | "
        f"PR-AUC: {metrics['PR-AUC']:.4f}"
    )
    return {
        "method_name": name,
        **metrics,
    }


def save_results_csv(
    rows: list[dict[str, float | str]],
    dataset_name: str,
    filename: str,
    root_dir: str = "results",
    subdirs: list[str] | None = None,
) -> Path:
    output_dir = Path(root_dir) / _sanitize_path_component(dataset_name)
    for part in subdirs or []:
        output_dir = output_dir / _sanitize_path_component(part)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{_sanitize_path_component(filename)}.csv"
    fieldnames = ["method_name", "ROC-AUC", "PR-AUC"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    print(f"Saved results to {output_path}")
    return output_path


def _scores_to_topk_predictions(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(y_true, dtype=np.int64)
    n_pos = int(np.sum(y_true == 1))

    if n_pos <= 0:
        return pred
    if n_pos >= len(scores):
        pred[:] = 1
        return pred

    top_idx = np.argpartition(scores, -n_pos)[-n_pos:]
    pred[top_idx] = 1
    return pred


def _sanitize_path_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"
