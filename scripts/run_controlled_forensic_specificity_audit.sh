#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${RIFT_CONTROLLED_AUDIT_CONFIG:-configs/controlled_forensic_audit/audit.yaml}"
STAGE="all"

if [[ $# -gt 0 ]]; then
  case "$1" in
    validate|calibrate|audit|aggregate|report|all)
      STAGE="$1"
      shift
      ;;
  esac
fi

python -m controlled_forensic_audit.specificity_audit.cli \
  --config "${CONFIG}" \
  --stage "${STAGE}" \
  "$@"
