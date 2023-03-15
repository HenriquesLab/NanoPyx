#!/bin/bash
export RUNNER_ALLOW_RUNASROOT="1"
apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
apt install -y curl systemctl build-essential lsb-release python3.11-full python3.10-full python3.10-venv python3-pip git libgl1-mesa-glx libglib2.0-0 ca-certificates jq htop gource ffmpeg xvfb libavcodec libpython3.10-dev libpython3.11-dev
python -m pip install --upgrade pip
python -m pip install --upgrade pipx
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.302.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.302.1/actions-runner-linux-x64-2.302.1.tar.gz
tar xzf ./actions-runner-linux-x64-2.302.1.tar.gz
./bin/installdependencies.sh
# call the commands to setup runner
./svc.sh install
./svc.sh start

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
curl micro.mamba.pm/install.sh | bash
micromamba install -y -c conda-forge pocl
conda install -y -c conda-forge pocl
