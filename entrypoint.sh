#!/usr/bin/env bash
set -e
# allow for overriding command via docker run CMD
# default behavior: run training script with config
if [ $# -eq 0 ]; then
    python train.py --config config.yaml
else
    exec "$@"
fi