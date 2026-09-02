#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

python3.10 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}"

echo "Created ${VENV_DIR}. Source /opt/ros/humble/setup.bash before using it."
