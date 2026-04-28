#!/bin/bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y git jq tmux htop python3-pip

mkdir -p /opt/arch1
chown ubuntu:ubuntu /opt/arch1
