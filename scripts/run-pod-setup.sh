#!/usr/bin/env bash

set -e

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv sync --extra cu124

git config --global user.email "tomasjanicek221@gmail.com"
git config --global user.name Tomas Janicek

apt-get update -y
apt-get install unzip screen nvtop htop -y


cp .env.example .env
echo -e "NUM_WORKERS=32" >> .env
echo -e "CACHE_DIR=/workspace/cache" >> .env
echo -e "DATA_DOWNLOADED_DIR=/workspace" >> .env
echo -e "LOG_EVERY_N_EXAMPLES=3000" >> .env

cp .env.tune.example .env.tune
echo -e "NUM_WORKERS=32" >> .env.tune
echo -e "CACHE_DIR=/workspace/cache" >> .env.tune
echo -e "DATA_DOWNLOADED_DIR=/workspace" >> .env.tune
