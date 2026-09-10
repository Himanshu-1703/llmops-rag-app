#!/usr/bin/env bash
# BeforeInstall -- installs Docker Engine and the AWS CLI v2, the same commands
# as deploy/ec2/manual-steps.md sections 1-3 (minus sudo -- this runs as root).
# Each install is skipped when it's already there, so re-deploys are quick.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive   # never stop for an apt prompt

APP_DIR=/opt/campusx-rag
APP_USER=ubuntu

# 1. Install Docker Engine
if ! command -v docker >/dev/null; then
  apt-get update
  apt-get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# 2. Install AWS CLI v2
if ! command -v aws >/dev/null; then
  apt-get install -y unzip
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip -o awscliv2.zip
  ./aws/install
  rm -rf awscliv2.zip aws
fi

# 3. Create the app directory and let the ubuntu user run docker without sudo
mkdir -p "$APP_DIR"
chown "$APP_USER":"$APP_USER" "$APP_DIR"
usermod -aG docker "$APP_USER"
