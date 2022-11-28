import sys
import os
import time
import random
import yaml
import argparse
import uuid
import resource
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import SaGnnAGP
from utils import load_multi_inductive,muticlass_f1

print("os.cpu_count: ", os.cpu_count())

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
print(config)


features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset = load_multi_inductive(args.datastr, config["dataset"], config["agp"])
print(features_train.shape,features.shape)

checkpt_file = f'pretrained/{os.path.basename(args.config)}.pt'

model = SaGnnAGP(nagp=features_train.shape[2], nfeat=features_train.shape[1],nlayers=config["layer"],nhidden=config["hidden"],nclass=int(labels.max()) + 1,dropout=config["dropout"],bias = config["bias"]).cuda(config["dev"])
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
loss_fn = nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(features_train, labels[idx_train])
loader = Data.DataLoader(dataset=torch_dataset,batch_size=config["batch"],shuffle=True,num_workers=min(9,os.cpu_count()))

def train():
    model.train()
    loss_list = []
    time_epoch = 0

    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.cuda(config["dev"])
        batch_y = batch_y.cuda(config["dev"])
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_train.item())
    return np.mean(loss_list),time_epoch


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features[idx_val].cuda(config["dev"]))
        micro_val = muticlass_f1(output, labels[idx_val])
        return micro_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features[idx_test].cuda(config["dev"]))
        micro_test = muticlass_f1(output, labels[idx_test])
        return micro_test.item()

train_time = 0
bad_counter = 0
best = 0
best_epoch = 0
print("--------------------------")
print("Training...")
for epoch in range(config["epochs"]):
    loss_tra,train_ep = train()
    train_time+=train_ep

    val_acc=validate()
    if(epoch+1)%50== 0:
        print(f'Epoch:{epoch+1:02d},'
            f'Train_loss:{loss_tra:.3f}',
            f'Valid_acc:{100*val_acc:.2f}%',
            f'Time_cost{train_time:.3f}')
    if val_acc > best:
        best = val_acc
        best_epoch = epoch+1
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == config["patience"]:
        break

test_acc = test()
print(f"Train cost: {train_time:.2f}s")
print('Load {}th epoch'.format(best_epoch))
print(f"Test accuracy:{100*test_acc:.2f}%")

memory_main = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
memory=memory_main-memory_dataset
print("Memory overhead:{:.2f}GB".format(memory))
print("--------------------------")
