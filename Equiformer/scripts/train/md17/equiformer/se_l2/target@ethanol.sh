#!/bin/bash

module load anaconda/2021.11
nvidia-smi
export PYTHONNOUSERSITE=True
source activate Equiformer
#cd ZKM/MLIP/NequIP/nequip-main
module unload compilers/cuda
module unload cudnn
source /home/bingxing2/apps/package/pytorch/backup_bug/1.10.0+cu113_cp38/env.sh
module list
python test_torch.py

python main_md17.py \
    --output-dir 'models/md17/equiformer/se_l2/target@ethanol/lr@5e-4_wd@1e-6_epochs@1500_w-f2e@80_dropout@0.0_exp@32_l2mae-loss' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --target 'ethanol' \
    --data-path 'datasets/md17' \
    --epochs 1500 \
    --lr 5e-4 \
    --batch-size 8 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 80
