#!/usr/bin/env bash
set -euo pipefail
python -m rift.validate --config configs/validate_cift.yaml "$@"
