import time
import uuid
import random
import argparse
import gc
import torch
import resource
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ogb.nodeproppred import Evaluator
from utils import SimpleDataset
from model import SaMLP
from utils import load_multi_transductive
import yaml

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/reddit_default.yml', help='config path.')
parser.add_argument('--datastr', default="data")
args = parser.parse_args()

with open(args.config, 'r') as yml:
    config = yaml.safe_load(yml)
    if "dev" not in config.keys():
        config["dev"] = None

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])

print("--------------------------")
print(args)
checkpt_file = f'pretrained/{os.path.basename(args.config)}.pt'

features,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset = load_multi_transductive(args.datastr, config["dataset"], config["agp"])

features_train = features[train_idx]
features_val = features[val_idx]
features_test = features[test_idx]
del features
gc.collect()

label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))+1
train_dataset = SimpleDataset(features_train,train_labels)
valid_dataset = SimpleDataset(features_val,val_labels)
test_dataset = SimpleDataset(features_test, test_labels)

train_loader = DataLoader(train_dataset, batch_size=config["batch"],shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = SaMLP(nagp=features_train.shape[2], in_channels=features_train.shape[1], hidden_channels=config["hidden"],out_channels=label_dim,num_layers=config["layer"],dropout=config["dropout"]).cuda(config["dev"])
evaluator = Evaluator(name='ogbn-papers100M')
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


def train(model, device, train_loader, optimizer):
	model.train()

	time_epoch=0
	loss_list=[]
	for step, (x, y) in enumerate(train_loader):
		t_st=time.time()
		x, y = x.cuda(device), y.cuda(device)
		optimizer.zero_grad()
		out = model(x)
		loss = F.nll_loss(out, y.squeeze(1))
		loss.backward()
		optimizer.step()
		time_epoch+=(time.time()-t_st)
		loss_list.append(loss.item())
	return np.mean(loss_list),time_epoch


@torch.no_grad()
def validate(model, device, loader, evaluator):
	model.eval()
	y_pred, y_true = [], []
	for step,(x,y) in enumerate(loader):
		x = x.cuda(device)
		out = model(x)
		y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
		y_true.append(y)
	return evaluator.eval({
		"y_true": torch.cat(y_true, dim=0),
		"y_pred": torch.cat(y_pred, dim=0),
	})['acc']


@torch.no_grad()
def test(model, device, loader, evaluator,checkpt_file):
	model.load_state_dict(torch.load(checkpt_file))
	model.eval()
	y_pred, y_true = [], []
	for step,(x,y) in enumerate(loader):
		x = x.cuda(device)
		out = model(x)
		y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
		y_true.append(y)
	return evaluator.eval({
		"y_true": torch.cat(y_true, dim=0),
		"y_pred": torch.cat(y_pred, dim=0),
	})['acc']

bad_counter = 0
best = 0
best_epoch = 0
train_time = 0
model.reset_parameters()
print("--------------------------")
print("Training...")
for epoch in range(config["epochs"]):
	loss_tra,train_ep = train(model,config["dev"],train_loader,optimizer)
	f1_val = validate(model, config["dev"], valid_loader, evaluator)
	train_time+=train_ep
	if(epoch+1)%20 == 0:
		print(f'Epoch:{epoch+1:02d},'
			f'Train_loss:{loss_tra:.3f}',
			f'Valid_acc:{100*f1_val:.2f}%',
			f'Time_cost{train_time:.3f}')
	if f1_val > best:
		best = f1_val
		best_epoch = epoch+1
		torch.save(model.state_dict(), checkpt_file)
		bad_counter = 0
	else:
		bad_counter += 1

	if bad_counter == config["patience"]:
		break

test_acc = test(model, config["dev"], test_loader, evaluator,checkpt_file)
print(f"Train cost: {train_time:.2f}s")
print('Load {}th epoch'.format(best_epoch))
print(f"Test accuracy:{100*test_acc:.2f}%")

memory_main = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
memory=memory_main-memory_dataset
print("Memory overhead:{:.2f}GB".format(memory))
print("--------------------------")
