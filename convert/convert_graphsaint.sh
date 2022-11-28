# pjsub --interact -j -g gs54 -L rscgrp=prepost -o ~/log.out

datastr="/data/scratch/gs54/s54002/data"
dataset="reddit"  #"reddit", "yelp"
#seed=0

source venv/bin/activate # ← 仮想環境を activate
cp -r classification-GNN/data/$dataset $datastr/$dataset/
cd classification-GNN/convert/
python convert_graphsaint.py --datastr $datastr --name $dataset
