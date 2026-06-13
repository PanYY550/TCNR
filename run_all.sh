#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${ENV_NAME}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # Ensure `conda activate` works in non-interactive shells.
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
  else
    echo "ERROR: conda not found, but env name '${ENV_NAME}' was provided." >&2
    exit 1
  fi
fi

run_step() {
  local workdir="$1"
  local script="$2"
  echo
  echo "==> [$(date +'%F %T')] ${workdir}/${script}"
  (cd "${ROOT_DIR}/${workdir}" && "${PYTHON_BIN}" "${script}")
}

run_step "Recall" "Recall_itemcf.py"
run_step "Recall" "DSSM_recall.py"
run_step "Recall" "Recall_merge.py"

run_step "Rank" "Feat_Eng.py"
run_step "Rank" "DCN_rank.py"
run_step "Rank" "din_rank.py"

echo
echo "All steps completed."
