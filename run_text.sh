#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TASK="${TASK:-}"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  TASK="$1"
  shift
fi

TASK_ARGS=()
if [ -n "${TASK}" ]; then
  case "${TASK}" in
    sms|sms_spam|"SMS spam classification")
      TASK_ARGS=(--task "SMS spam classification")
      ;;
    ag|ag_news|"AG News Classification")
      TASK_ARGS=(--task "AG News Classification")
      ;;
    email|email_spam|"To check if an email is a spam")
      TASK_ARGS=(--task "To check if an email is a spam")
      ;;
    yelp|"Yelp reviews dataset consists of reviews from Yelp")
      TASK_ARGS=(--task "Yelp reviews dataset consists of reviews from Yelp")
      ;;
    *)
      echo "Unknown task '${TASK}'." >&2
      echo "Choose one of: sms, ag_news, email_spam, yelp." >&2
      exit 2
      ;;
  esac
fi

echo "Running NLP-ADBench MDAD parametric text REC"
python -m experiments.run_mdad_parametric_rec_text \
  "${TASK_ARGS[@]}" \
  "$@"
