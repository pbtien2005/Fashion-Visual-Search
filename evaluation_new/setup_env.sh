#!/usr/bin/env bash
# Create the conda env `fvs-bench` for evaluation_new/. Does not touch conda base.
#   bash evaluation_new/setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-fvs-bench}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
PY=(conda run --no-capture-output -n "$ENV_NAME" python)
"${PY[@]}" -m pip install --upgrade pip
"${PY[@]}" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"
"${PY[@]}" -m pip install -r "$HERE/requirements.txt"
"${PY[@]}" -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
"${PY[@]}" "$HERE/check_env.py"
