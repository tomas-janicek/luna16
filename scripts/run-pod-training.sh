#!/usr/bin/env bash

set -e

export PYTHONPATH=.

uv run python luna16/cli.py \
              train_luna_classification \
              1.0.0 \
              --epochs=10 \
              --batch-size=64
