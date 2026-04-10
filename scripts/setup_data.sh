#!/usr/bin/env bash
set -euo pipefail

HPC_DATA_PATH="/dtu/blackhole/02/137570/MML"
LINK_TARGET="$(pwd)/data/MML_Data"

if [ ! -d "$HPC_DATA_PATH" ]; then
  echo "ERROR: $HPC_DATA_PATH not found. Run this on HPC." >&2
  exit 1
fi

if [ -L "$LINK_TARGET" ]; then
  echo "Symlink already exists: $LINK_TARGET -> $(readlink "$LINK_TARGET")"
  exit 0
fi

mkdir -p data
ln -s "$HPC_DATA_PATH" "$LINK_TARGET"
echo "Created: $LINK_TARGET -> $HPC_DATA_PATH"
