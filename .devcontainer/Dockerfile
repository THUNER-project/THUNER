FROM debian:12-slim

ARG USERNAME=thuner
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG PIXI_VERSION=0.69.0

ENV DEBIAN_FRONTEND=noninteractive

# System tools, fonts (matplotlib), and Node.js 20 (for @anthropic-ai/claude-code)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo less man-db gnupg2 bzip2 xz-utils \
        ripgrep fd-find \
        fonts-liberation fonts-dejavu fonts-freefont-ttf \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with passwordless sudo
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install pixi system-wide. Pinned via build arg; bump to update.
RUN curl -fsSL "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
        | tar -xz -C /usr/local/bin pixi \
    && chmod +x /usr/local/bin/pixi

ENV NPM_CONFIG_PREFIX=/home/$USERNAME/.npm-global
ENV PATH=$NPM_CONFIG_PREFIX/bin:$PATH

USER $USERNAME
RUN mkdir -p $NPM_CONFIG_PREFIX \
    && npm install -g @anthropic-ai/claude-code

WORKDIR /workspace
