#!/bin/sh -l

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -g gs54
#PJM -j
#PJM -m e

module load cuda/11.1

source venv/bin/activate # ← 仮想環境を activate
cd classification-GNN/
python -u papers100M.py --dataset papers100M --agp_alg appnp_agp --alpha 0.1 --L 20 --rmax 1e-7 --lr 0.0001 --dropout 0.3 --hidden 2048
