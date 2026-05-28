#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET_ARGS=()
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  DATASET_ARGS=(--dataset "$1")
  shift
fi

echo "Running tabular MDAD nonparametric REC"
python -m experiments.run_mdad_nonparametric_rec \
  "${DATASET_ARGS[@]}" \
  "$@"

echo
echo "Running tabular MDAD parametric REC"
python -m experiments.run_mdad_parametric_rec \
  "${DATASET_ARGS[@]}" \
  "$@"
