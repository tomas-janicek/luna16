#!/usr/bin/env bash

set -e

export PYTHONPATH=.

uv run python luna16/cli.py train_luna_classification --epochs=10 \
                                                      --batch-size=256 \
                                                      --profile 
