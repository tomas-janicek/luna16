#!/usr/bin/env bash

set -e

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv sync

git config --global user.email "tomasjanicek221@gmail.com"
git config --global user.name Tomas Janicek

apt-get update
apt-get install unzip

echo -e >> .env
echo -e "NUM_WORKERS=64" >> .env
echo -e "CACHE_DIR=/workspace/cache" >> .env
echo -e "DATA_DOWNLOADED_DIR=/workspace/data_downloaded" >> .env
