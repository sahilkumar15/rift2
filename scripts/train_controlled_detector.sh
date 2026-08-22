#!/usr/bin/env bash

set -euo pipefail


# ----------------------------------------------------------------------
# Repository
# ----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"


# ----------------------------------------------------------------------
# Python source
# ----------------------------------------------------------------------

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1


# ----------------------------------------------------------------------
# Configuration
#
# All experiment/training parameters are controlled by this YAML:
#
#   - experiment name
#   - dataset
#   - batch size
#   - learning rate
#   - max epochs
#   - early stopping
#   - checkpointing
#   - resume
#   - W&B
#   - shortcut parameters
#   - etc.
# ----------------------------------------------------------------------

CONFIG="${RIFT_CONTROLLED_DETECTOR_CONFIG:-configs/controlled_forensic_audit/detector.yaml}"


# ----------------------------------------------------------------------
# GPU selection
#
# Usage:
#
#   ./scripts/train_controlled_detector.sh 6,7,2,0
#
#   ./scripts/train_controlled_detector.sh 2,0
#
#   ./scripts/train_controlled_detector.sh 6
#
# If no GPU argument is supplied, use RIFT_GPUS when defined.
# Otherwise default to 6,7,2,0.
# ----------------------------------------------------------------------

GPUS="${1:-${RIFT_GPUS:-6,7,2,0}}"

export CUDA_VISIBLE_DEVICES="${GPUS}"


# ----------------------------------------------------------------------
# Basic checks
# ----------------------------------------------------------------------

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found:"
    echo "        ${CONFIG}"
    exit 1
fi


# ----------------------------------------------------------------------
# Run information
# ----------------------------------------------------------------------

echo
echo "========================================================================"
echo "CONTROLLED FORENSIC DETECTOR TRAINING"
echo "========================================================================"
echo "Repository : ${REPO_ROOT}"
echo "Config     : ${CONFIG}"
echo "GPUs       : ${CUDA_VISIBLE_DEVICES}"
echo "Module     : controlled_forensic_audit.train"
echo "========================================================================"
echo


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

python -m controlled_forensic_audit.train \
    --config "${CONFIG}"


#  bash scripts/train_controlled_detector.sh 6,7,2,1