#!/bin/bash

module load anaconda/2021.11
nvidia-smi
export PYTHONNOUSERSITE=True
source activate equiformer_torch2
#cd ZKM/MLIP/NequIP/nequip-main
module unload compilers/cuda
module unload cudnn
source /home/bingxing2/apps/package/pytorch/2.1.2+cu118_py310/env.sh
module list
python test_torch.py

python main_tobermorite.py \
    --output-dir 'models/CSH/equiformer/se_l3/target@tob11_950-50/lr@2e-4_bs@4_wd@1e-6_epochs@1000_w-f2e@100_dropout@0.0_exp@32_l2mae-loss' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l3_md17' \
    --input-irreps '64x0e' \
    --target 'tob11' \
    --data-path 'datasets/tobermorite' \
    --epochs 1000 \
    --radius 4.0 \
    --train-size 950 \
    --val-size 50 \
    --lr 2e-4 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 100
