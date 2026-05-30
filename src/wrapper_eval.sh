#!/bin/bash
# =====================================================
#  HTCondor wrapper for AD evaluation
# =====================================================

set -euo pipefail

echo "[$(date)] Starting Condor job on $(hostname)"
echo "[$(date)] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/Baseline_AD_Collide2v
PYTHON=/eos/user/d/dgenoves/conda_envs/collidenv/bin/python3
CKPT_PATH="${AD_CKPT_PATH:-}"

cd ${PROJECT_DIR}

mkdir -p ${PROJECT_DIR}/logs/condor

if [ -z "${CKPT_PATH}" ]; then
    echo "ERROR: AD_CKPT_PATH is not set. Set CKPT_PATH in eval.sub."
    exit 1
fi

echo "[$(date)] Evaluating checkpoint: ${CKPT_PATH}"

${PYTHON} src/eval.py ckpt_path=${CKPT_PATH}

EXIT_CODE=$?
echo "[$(date)] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
