#!/usr/bin/env bash

set -e

export PYTHONPATH=.

uv run python luna16/cli.py \
              train_luna_classification \
              0.0.1-profile \
              --epochs=10 \
              --batch-size=64 \
              --profile 
