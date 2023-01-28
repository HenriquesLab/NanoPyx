#!/bin/bash
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfile_py311 -t henriqueslab/nanopyx-gha-runner-py311:latest .
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfile_py310 -t henriqueslab/nanopyx-gha-runner-py310:latest .
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfile_py39 -t henriqueslab/nanopyx-gha-runner-py39:latest .
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfile_py38 -t henriqueslab/nanopyx-gha-runner-py38:latest .