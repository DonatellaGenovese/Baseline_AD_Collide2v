#!/bin/bash
# =====================================================
#  HTCondor wrapper for AD training
# =====================================================

set -euo pipefail

echo "[$(date)] Starting Condor job on $(hostname)"
echo "[$(date)] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/Baseline_AD_Collide2v
PYTHON=/eos/user/d/dgenoves/conda_envs/collidenv/bin/python3
EXPERIMENT="${AD_EXPERIMENT:-ad_ae}"

cd ${PROJECT_DIR}

# Create condor log dir if needed
mkdir -p ${PROJECT_DIR}/logs/condor

echo "[$(date)] Running experiment: ${EXPERIMENT}"

${PYTHON} src/train.py experiment=${EXPERIMENT}

EXIT_CODE=$?
echo "[$(date)] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
