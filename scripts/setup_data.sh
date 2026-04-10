#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HPC_DATA_PATH="/dtu/blackhole/02/137570/MML"
LINK_TARGET="$REPO_ROOT/data/MML_Data"
EDA_NOTEBOOK="$REPO_ROOT/notebooks/team/02_eda.ipynb"

# --- Symlink setup ---
if [ ! -d "$HPC_DATA_PATH" ]; then
  echo "ERROR: $HPC_DATA_PATH not found. Run this on HPC." >&2
  exit 1
fi

if [ -L "$LINK_TARGET" ]; then
  echo "Symlink already exists: $LINK_TARGET -> $(readlink "$LINK_TARGET")"
else
  mkdir -p "$REPO_ROOT/data"
  ln -s "$HPC_DATA_PATH" "$LINK_TARGET"
  echo "Created: $LINK_TARGET -> $HPC_DATA_PATH"
fi

# --- Run EDA notebook ---
if [ ! -f "$EDA_NOTEBOOK" ]; then
  echo "ERROR: Notebook not found: $EDA_NOTEBOOK" >&2
  exit 1
fi

echo "Running EDA notebook..."
uv run jupyter nbconvert --to notebook --execute --inplace "$EDA_NOTEBOOK"
echo "EDA notebook executed successfully."
