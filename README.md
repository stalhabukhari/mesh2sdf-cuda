# CUDA-accelerated Signed Distance queries from Meshes

## Setup

Tested with CUDA 11.7 (ensure compatible PyTorch version in `pkgs-install.sh`)

```sh
conda create -y -n m2s python=3.8
conda activate m2s
source pkgs-install.sh
```

## Run

Example: Given `$DATA_DIR` contains meshes as `.obj` files

```sh
cd src/
python main.py \
    --dataset_dir $DATADIR \
    --save_dir $SAVEDIR \
    --num_samples_surf 100000 \
    --num_samples_sdf 600000 \
    --chunk_size 100000
```

## Acknowledgements

Code in this repository is adapted from:
- https://github.com/nv-tlabs/nglod
- https://github.com/zekunhao1995/DualSDF
