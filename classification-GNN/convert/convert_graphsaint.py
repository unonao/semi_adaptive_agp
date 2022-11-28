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
import argparse
from torch_geometric.utils import homophily
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torch

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

def load_data(dataset_path,prefix, normalize=True):
    adj_full = scipy.sparse.load_npz('{}/{}/adj_full.npz'.format(dataset_path,prefix)).astype(bool)
    num_edges = adj_full.nonzero()[0].shape[0]
    adj_train = scipy.sparse.load_npz('{}/{}/adj_train.npz'.format(dataset_path,prefix)).astype(bool)
    role = json.load(open('{}/{}/role.json'.format(dataset_path,prefix)))
    feats = np.load('{}/{}/feats.npy'.format(dataset_path,prefix))
    class_map = json.load(open('{}/{}/class_map.json'.format(dataset_path,prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    node_train = np.array(role['tr'])
    node_val = np.array(role['va'])
    node_test = np.array(role['te'])
    train_feats = feats[node_train]
    adj_train = adj_train[node_train,:][:,node_train]
    labels = class_arr

    adj_train=adj_train+sp.eye(adj_train.shape[0])
    adj_full=adj_full+sp.eye(adj_full.shape[0])


    num_edges_self = adj_full.nonzero()[0].shape[0]
    num_features = feats.shape[1]



    stats_str = f"""
    nodes:{num_vertices}, edges:{num_edges}, edges w/ self-loop:{num_edges_self}, features:{num_features}, classes:{num_classes}, train:{len(node_train)}, val:{len(node_val)}, test:{len(node_test)}
    """
    if prefix=='yelp':
        print('multi label')
        print(stats_str) #
    elif prefix=='reddit':
        print('multi class')
        labels=np.where(labels>0.5)[1] # multiclass用
        # homophily
        A = adj_full.tocoo()
        row = torch.from_numpy(A.row).to(torch.long)
        col = torch.from_numpy(A.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        stats_str += f", node homophily:{homophily(edge_index, torch.tensor(labels), method='node')}"
        stats_str += f", edge homophily:{homophily(edge_index, torch.tensor(labels), method='edge')}"
        stats_str += f", edge_insensitive homophily:{homophily(edge_index, torch.tensor(labels), method='edge_insensitive')}"
        print(stats_str)


    return adj_full, adj_train, feats, train_feats, labels, node_train, node_val, node_test

def graphsaint(datastr,dataset_name):
    if dataset_name=='yelp': # multilabel
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr,'yelp')
        graphsave(adj_full,dir='../data/yelp_full_adj_')
        graphsave(adj_train,dir='../data/yelp_train_adj_')
        feats=np.array(feats,dtype=np.float64)
        train_feats=np.array(train_feats,dtype=np.float64)
        np.save(os.path.join(datastr, f'{dataset_name}_feat.npy'),feats)
        np.save(os.path.join(datastr, f'{dataset_name}_train_feat.npy'),feats)
        np.savez(os.path.join(datastr, f'{dataset_name}_labels.npz'),labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)
    if dataset_name=='reddit': # multiclass用
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr,'reddit')
        graphsave(adj_full,dir=os.path.join(datastr, f'{dataset_name}_full_adj_'))
        graphsave(adj_train,dir=os.path.join(datastr, f'{dataset_name}_train_adj_'))
        feats=np.array(feats,dtype=np.float64)
        train_feats=np.array(train_feats,dtype=np.float64)
        np.save(os.path.join(datastr, f'{dataset_name}_feat.npy'),feats)
        np.save(os.path.join(datastr, f'{dataset_name}_train_feat.npy'),feats)
        np.savez(os.path.join(datastr, f'{dataset_name}_labels.npz'),labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)

if __name__ == "__main__":
    #Your file storage path. For example, this is shown below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--datastr', default="../data")
    parser.add_argument('--name', default="yelp") #yelp or reddit
    args = parser.parse_args()
    print(args.datastr, args.name)
    graphsaint(args.datastr,args.name)
