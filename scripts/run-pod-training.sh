#!/usr/bin/env bash

set -e

export PYTHONPATH=.

uv run python luna16/cli.py train_luna_classification --epochs=1 \
                                                      --batch-size=64
