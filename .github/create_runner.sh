#!/bin/bash
export RUNNER_ALLOW_RUNASROOT="1"
apt update
apt install -y curl systemctl build-essential lsb-release python3-pip git libgl1-mesa-glx libglib2.0-0 ca-certificates jq htop gource ffmpeg xvfb python3.10-venv
# part 2
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.302.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.302.1/actions-runner-linux-x64-2.302.1.tar.gz
tar xzf ./actions-runner-linux-x64-2.302.1.tar.gz
./bin/installdependencies.sh
# call the commands to setup runner
./svc.sh install
./svc.sh start
