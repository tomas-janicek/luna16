#!/usr/bin/env bash

set -e

cp .env.example .env

mv uv.lock uv.cpu.lock
mv uv.lock.cuda uv.lock

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.cargo/env 

uv sync

git config --global user.email "tomasjanicek221@gmail.com"
git config --global user.name Tomas Janicek

apt-get update
apt-get install unzip
