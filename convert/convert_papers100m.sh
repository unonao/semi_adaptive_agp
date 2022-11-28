datastr="/data/scratch/gs54/s54002/data"
#seed=0

source venv/bin/activate # ← 仮想環境を activate
cd classification-GNN/convert/
yes | python convert_papers100M.py --datastr $datastr
