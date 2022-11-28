# Planetoid #
# https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html?highlight=CiteSeer#torch_geometric.datasets.Planetoid

import pickle as pkl
import sys
import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
import json
import time
import scipy.sparse
import struct
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork,Actor,WebKB
from torch_geometric.utils import homophily
import torch
import random
import argparse


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0, seed=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    torch.manual_seed(seed)
    random.seed(seed)

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def graphsave(adj,dir):
    if(sp.isspmatrix_csr(adj)):
        el=adj.indices
        pl=adj.indptr
        EL=np.array(el,dtype=np.uint32)
        PL=np.array(pl,dtype=np.uint32)
        EL_re=[]
        for i in range(1,PL.shape[0]):
            EL_re+=sorted(EL[PL[i-1]:PL[i]],key=lambda x:PL[x+1]-PL[x])
        EL_re=np.asarray(EL_re,dtype=np.uint32)
        print("EL:",EL_re.shape)
        f1=open(dir+'el.txt','wb')
        for i in EL_re:
            m=struct.pack('I',i)
            f1.write(m)
        f1.close()
        print("PL:",PL.shape)
        f2=open(dir+'pl.txt','wb')
        for i in PL:
            m=struct.pack('I',i)
            f2.write(m)
        f2.close()
    else:
        print("Format Error!")


def load_data(datastr, dataset_name, seed=0):
    # homophilic (2.5%/2.5%/95%)
    if dataset_name in ["Cora", "CiteSeer", "PubMed","Computers", "Photo"]:
        train_rate = 0.025
        val_rate = 0.025
        if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
            dataset = Planetoid(root=datastr, name=dataset_name) # split = "geom-gcn"
        else:
            dataset = Amazon(root=datastr, name=dataset_name)
        data = dataset[0]
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, seed=seed)
    # heterophic (60%/20%/20%)
    elif dataset_name in ["chameleon", "squirrel","Actor","Texas", "Cornell"]:
        train_rate = 0.60
        val_rate = 0.20
        if dataset_name in ["chameleon", "squirrel"]:
            dataset = WikipediaNetwork(root=datastr, name=dataset_name)
        elif dataset_name in ["Actor"]:
            dataset = Actor(root=datastr)
        elif dataset_name in ["Texas", "Cornell"]:
            dataset = WebKB(root=datastr, name=dataset_name)
        data = dataset[0]
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, seed=seed)
    else:
        raise "dataset name error"

    num_vertices =  data.num_nodes
    num_edges = data.num_edges
    num_features = data.num_node_features
    num_classes = data.y.max() - data.y.min() + 1

    idx_train =  data.train_mask.nonzero().flatten().numpy()
    idx_val =  data.val_mask.nonzero().flatten().numpy()
    idx_test =  data.test_mask.nonzero().flatten().numpy()
    feats = data.x.numpy()
    train_feats = feats[idx_train]
    # scalling
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    train_feats = feats[idx_train]
    # labels
    """
    y = data.y.numpy()
    class_arr = np.zeros((num_vertices, num_classes))
    offset = y.min()
    for k,v in enumerate(y):
        class_arr[k][v-offset] = 1
    labels = class_arr
    """
    labels = data.y.numpy()
    def _construct_adj(edges):
        adj = sp.csr_matrix((np.ones(
            (edges.shape[1]), dtype=np.float32), (edges[0, :], edges[1, :])),
                            shape=(num_vertices, num_vertices))
        # adj += adj.transpose()
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj=adj+sp.eye(adj.shape[0])
        return adj
    adj_full = _construct_adj(data.edge_index.numpy())
    num_edges_self = adj_full.nonzero()[0].shape[0]

    print(f"nodes:{num_vertices}, edges:{num_edges}, edges w/ self-loop:{num_edges_self}, features:{num_features}, classes:{num_classes}, train:{len(idx_train)}, val:{len(idx_val)}, test:{len(idx_test)}, node homophily:{homophily(data.edge_index, data.y, method='node')}, edge homophily:{homophily(data.edge_index, data.y, method='edge')}, edge_insensitive homophily:{homophily(data.edge_index, data.y, method='edge_insensitive')}")

    return adj_full, feats, train_feats, labels, idx_train, idx_val, idx_test,num_vertices,num_edges_self

def convert_geometric(datastr,dataset_name,seed):
    adj_full, feats, train_feats, labels, idx_train, idx_val, idx_test, num_vertices,num_edges = load_data(datastr,dataset_name, seed)
    graphsave(adj_full,dir=os.path.join(datastr, f'{dataset_name}_adj_')) # agpAlgで読み込まれる
    feats=np.array(feats,dtype=np.float64)
    train_feats=np.array(train_feats,dtype=np.float64)
    np.save(os.path.join(datastr, f'{dataset_name}_feat.npy'),feats)
    np.save(os.path.join(datastr, f'{dataset_name}_train_feat.npy'),train_feats)
    np.savez(os.path.join(datastr, f'{dataset_name}_labels.npz'),labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test,num_vertices=num_vertices,num_edges=num_edges)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datastr', default="../data")
    parser.add_argument('--name', default="Cora")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(args.datastr, args.name, args.seed)
    convert_geometric(args.datastr,args.name,args.seed)
