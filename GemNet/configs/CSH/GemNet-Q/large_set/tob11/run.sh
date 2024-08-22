#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2021.11
nvidia-smi
#export PATH=/home/bingxing2/home/scx6d1j/.local/bin:$PATH
source activate gemnet
#cd ZKM/MLIP/NequIP/nequip-main
#pip install /home/bingxing2/apps/package/pytorch/1.10.1+cu111_cp38/*.whl
source /home/bingxing2/apps/package/pytorch/1.12.1+cu116_cp310/env.sh
module list
python test_torch.py
#watch -n 60 nvidia-smi
python train.py