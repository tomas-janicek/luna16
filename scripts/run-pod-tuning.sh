#!/usr/bin/env bash

set -e

export PYTHONPATH=.

mv .env .env.backup
mv .env.tune .env

source .env

uv run python luna16/cli.py \
              tune_luna_classification \
              --epochs=10

mv .env .env.tune
mv. env.backup .env
