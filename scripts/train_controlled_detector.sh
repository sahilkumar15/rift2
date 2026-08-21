#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${RIFT_CONTROLLED_DETECTOR_CONFIG:-configs/controlled_forensic_audit/detector.yaml}"
python -m controlled_forensic_audit.train --config "${CONFIG}" "$@"
