#!/usr/bin/env bash

set -e

mv .env .env.backup
cp .env.tune .env
echo -e "NUM_WORKERS=10" >> .env
echo -e "CACHE_DIR=/workspace/cache" >> .env
echo -e "DATA_DOWNLOADED_DIR=/workspace" >> .env


export PYTHONPATH=.

uv run python luna16/cli.py \
              tune_luna_classification \
              --epochs=1
