#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2021.11
nvidia-smi
#export PATH=/home/bingxing2/home/scx6d1j/.local/bin:$PATH
source activate spherenet
#cd ZKM/MLIP/NequIP/nequip-main
#pip install /home/bingxing2/apps/package/pytorch/1.10.1+cu111_cp38/*.whl
source /home/bingxing2/apps/package/pytorch/2.1.2+cu118_py310/env.sh
module list
python test_torch.py
python SphereNet.py