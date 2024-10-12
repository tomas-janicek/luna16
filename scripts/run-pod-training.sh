#!/usr/bin/env bash

set -e

uv run python luna16/cli.py train_luna_classification --epochs=10 \
                                                      --batch-size=64
