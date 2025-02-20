#!/usr/bin/env bash

set -e

cp .env .env.backup
cp .env.tune .env

echo -e "NUM_WORKERS=10" >> .env
echo -e "CACHE_DIR=/workspace/cache" >> .env

export PYTHONPATH=.

uv run python luna16/cli.py \
              tune_luna_classification \
              --epochs=10

mv .env.backup .env
rm .env.backup
