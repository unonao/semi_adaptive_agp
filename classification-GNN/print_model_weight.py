import os
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/cora_appnp_0.1_0.9.yml', help='config path.')
args = parser.parse_args()

checkpt_file = f'pretrained/{os.path.basename(args.config)}.pt'
print(checkpt_file)
model_data = torch.load(checkpt_file, map_location=torch.device('cpu'))
print(model_data['linear.weight'])
