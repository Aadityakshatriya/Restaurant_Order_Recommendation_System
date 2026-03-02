#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <owner/space-name> [public|private]"
  exit 1
fi

SPACE_ID="$1"
VISIBILITY="${2:-public}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
REPO_DIR="$TMP_DIR/space-repo"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install with: pip install -U huggingface_hub"
  exit 1
fi

WHOAMI_OUTPUT="$(hf auth whoami 2>&1 || true)"
if [[ -z "$WHOAMI_OUTPUT" || "$WHOAMI_OUTPUT" == *"Not logged in"* ]]; then
  echo "Not logged in. Run: hf auth login"
  exit 1
fi

TOKEN_ROLE="$(python - <<'PY'
from huggingface_hub import HfApi
try:
    role = HfApi().whoami().get("auth", {}).get("accessToken", {}).get("role", "")
except Exception:
    role = ""
print(role)
PY
)"
if [[ "$TOKEN_ROLE" == "read" ]]; then
  echo "Current HF token is read-only. Create/login with a write token, then retry."
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Missing git-lfs. Install git-lfs and retry."
  exit 1
fi

echo "Ensuring Space exists: $SPACE_ID"
CREATE_ARGS=(repo create "$SPACE_ID" --type space --space_sdk docker -y)
if [[ "$VISIBILITY" == "private" ]]; then
  CREATE_ARGS+=(--private)
fi
hf "${CREATE_ARGS[@]}" >/dev/null 2>&1 || true

echo "Cloning Space repo..."
git clone "https://huggingface.co/spaces/$SPACE_ID" "$REPO_DIR"

echo "Syncing project files..."
rsync -a --delete \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude "csao_logs/" \
  --exclude "csao_models/registry/" \
  --exclude "csao_models/lgbm_ranker_business.joblib" \
  --exclude "csao_models/feature_importances_business.csv" \
  --exclude ".venv/" \
  "$ROOT_DIR/" "$REPO_DIR/"

pushd "$REPO_DIR" >/dev/null
git lfs install
git add .

if git diff --cached --quiet; then
  echo "No changes to deploy."
  popd >/dev/null
  exit 0
fi

git commit -m "Deploy CSAO recommender space"
git push
popd >/dev/null

echo "Deployment pushed: https://huggingface.co/spaces/$SPACE_ID"
