#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT="/home/idsia/SonoGym"

SCRIPT_DIR="${REPO_ROOT}/source/spinal_surgery/spinal_surgery/tasks/cem_probe_optimize"
SCRIPT="${SCRIPT_DIR}/cem_probe_optimize.py"
BASE_CFG="${SCRIPT_DIR}/cem_probe_optimize_cfg.yaml"

# Logs/results root
RUN_GROUP_NAME="kuka_surgery_cem_probe_sweep"
BASE_OUT="${REPO_ROOT}/logs/skrl/CEM/${RUN_GROUP_NAME}"

CONFIG_DIR="${BASE_OUT}/configs"
LOG_DIR="${BASE_OUT}/logs"
RESULTS_DIR="${BASE_OUT}/results"

mkdir -p "${CONFIG_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# -----------------------------------------------------------------------------
# Seeds
# -----------------------------------------------------------------------------
SEEDS=(5 6 7 8)

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="kuka_surgery_cem_probe_seed${SEED}"

    OUT_DIR="${RESULTS_DIR}/seed_${SEED}"
    CFG_OUT="${CONFIG_DIR}/cem_probe_optimize_seed${SEED}.yaml"
    LOG_OUT="${LOG_DIR}/seed_${SEED}.log"

    mkdir -p "${OUT_DIR}"

    echo "============================================================"
    echo "Starting CEM run with seed ${SEED}"
    echo "Run name : ${RUN_NAME}"
    echo "Script   : ${SCRIPT}"
    echo "Config   : ${CFG_OUT}"
    echo "Output   : ${OUT_DIR}"
    echo "Log      : ${LOG_OUT}"
    echo "============================================================"

    python - <<PY
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

base_cfg = Path("${BASE_CFG}")
cfg_out = Path("${CFG_OUT}")

with base_cfg.open("r") as f:
    cfg = yaml.load(f)

cfg["seed"] = ${SEED}
cfg["output_dir"] = "${OUT_DIR}"

if "logging" not in cfg:
    cfg["logging"] = {}

cfg["logging"]["wandb_run_name"] = "${RUN_NAME}"

with cfg_out.open("w") as f:
    yaml.dump(cfg, f)
PY

    cd "${REPO_ROOT}"

    python "${SCRIPT}" \
        --cfg "${CFG_OUT}" \
        --run_mode play \
        --headless \
        --device cuda:0 \
        > "${LOG_OUT}" 2>&1

    echo "Finished seed ${SEED}"
    echo ""
done

echo "All CEM sweep runs completed."
echo "Saved in: ${BASE_OUT}"