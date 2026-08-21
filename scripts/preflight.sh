#!/usr/bin/env bash
set -euo pipefail
python -m rift.preflight --config configs/train_detector_mixed.yaml "$@"
