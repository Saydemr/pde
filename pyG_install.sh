#!/usr/bin/env bash
TORCH=1.11.0
CUDA=cu113  # Supply as command line cpu or cu102

pip install ogb pykeops -U --no-cache-dir
pip install torch==1.8.1 -U --no-cache-dir
pip install torchdiffeq==0.2.2 -U --no-cache-dir -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-scatter==2.0.9 -U --no-cache-dir  -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.12 -U --no-cache-dir  -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.9 -U --no-cache-dir  -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.1 -U --no-cache-dir  -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==2.0.3 -U  --no-cache-dir 
