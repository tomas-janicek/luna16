# TODO: This is not fully working yet.
# TODO: It must be tested and adjusted to other distros and architectures.
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
RUN echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc

RUN apt update -y
RUN apt install -y pkg-config
