#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Always run from project root, so IsaacLab/SKRL saves in the standard
# /home/idsia/SonoGym/logs/skrl/... location, not inside the task folder.
# ---------------------------------------------------------------------
PROJECT_ROOT="/home/idsia/SonoGym"
cd "${PROJECT_ROOT}"

TASK_DIR="${PROJECT_ROOT}/source/spinal_surgery/spinal_surgery/tasks/robot_US_guidance_Pink"

AGENT_CFG="${TASK_DIR}/agents/skrl_ppo_cfg.yaml"
TRAIN_SCRIPT="${PROJECT_ROOT}/workflows/skrl/train.py"

TASK_NAME="Isaac-robot-US-guidance-pink-v0"

SEEDS=(1 2 3 4 5)

# Backup temporaneo in /tmp, cancellato automaticamente
ORIGINAL_AGENT_CFG="$(mktemp)"
cp "${AGENT_CFG}" "${ORIGINAL_AGENT_CFG}"

restore_config() {
    echo "Restoring original agent config..."
    cp "${ORIGINAL_AGENT_CFG}" "${AGENT_CFG}"
    rm -f "${ORIGINAL_AGENT_CFG}"
}

trap restore_config EXIT

for SEED in "${SEEDS[@]}"; do
    EXP_NAME="PPO_guidance_seed${SEED}"

    echo "============================================================"
    echo "Starting guidance training run with seed ${SEED}"
    echo "Working dir : ${PROJECT_ROOT}"
    echo "Experiment  : ${EXP_NAME}"
    echo "============================================================"

    python - <<PY
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

cfg_path = Path("${AGENT_CFG}")

with cfg_path.open("r") as f:
    cfg = yaml.load(f)

cfg["seed"] = ${SEED}

cfg.setdefault("agent", {})
cfg["agent"].setdefault("experiment", {})

# IsaacLab/SKRL standard output:
# /home/idsia/SonoGym/logs/skrl/US_guidance_Pink/PPO_guidance_seedX/
cfg["agent"]["experiment"]["directory"] = "US_guidance_Pink"
cfg["agent"]["experiment"]["experiment_name"] = "${EXP_NAME}"

with cfg_path.open("w") as f:
    yaml.dump(cfg, f)
PY

    python "${TRAIN_SCRIPT}" \
        --task "${TASK_NAME}" \
        --headless \
        --device cuda:0

    echo "Finished guidance seed ${SEED}"
    echo ""
done

echo "All guidance sweep runs completed."