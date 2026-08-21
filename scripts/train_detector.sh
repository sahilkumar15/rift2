#!/usr/bin/env bash
set -euo pipefail
python -m rift.train --config configs/train_detector_mixed.yaml "$@"
