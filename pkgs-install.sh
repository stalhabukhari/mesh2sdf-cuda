#!/bin/bash

# env: m2s
pip install --upgrade pip && \
    conda install -y pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia && \
    pip install -r infra/requirements.txt && \
    pip install ninja && \
    cd src/lib/extensions/mesh2sdf_cuda && \
    python setup.py clean --all install && \
    cd ../../../..
