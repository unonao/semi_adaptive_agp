# pjsub --interact -j -g gs54 -L rscgrp=prepost -o ~/log.out

datastr="/data/scratch/gs54/s54002/data"
dataset="chameleon"  #"Cora", "CiteSeer", "PubMed","Computers", "Photo", "chameleon", "squirrel","Actor","Texas", "Cornell"
seed=0

source venv/bin/activate # ← 仮想環境を activate
cd classification-GNN/convert/
python convert_geometric.py --datastr $datastr --name $dataset --seed $seed
