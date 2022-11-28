import torch
import gc
import numpy as np
import copy
import os
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from propagation import AGP
import pickle
import time



def load_geometric_data(datastr, dataset_name, params_list):
	dataset_el = os.path.join(datastr,dataset_name+"_adj_el.txt")
	dataset_pl = os.path.join(datastr,dataset_name+"_adj_pl.txt")

	data = np.load(os.path.join(datastr,dataset_name+"_labels.npz"))
	labels = data['labels']
	idx_train = data['idx_train']
	idx_val = data['idx_val']
	idx_test = data['idx_test']
	labels = torch.LongTensor(labels)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)

	n = data["num_vertices"]
	m = data["num_edges"]

	py_agp=AGP()
	print("Load graph and initialize! It could take a few minutes...")

	raw_features = np.load(os.path.join(datastr,dataset_name+'_feat.npy'))
	features = np.tile(raw_features, (len(params_list), 1, 1)).transpose((1,2,0))
	del raw_features
	gc.collect()
	features = features.astype(np.float64, copy=False)
	print("features shape", features.shape)

	for i, params in enumerate(params_list):
		print(params)
		agp_alg = params['agp_alg']
		L = params['L']
		rmax = params['rmax']
		alpha = params['alpha']
		t = params['ti']
		memory_dataset= py_agp.agp_operation(dataset_el, dataset_pl, dataset_name,agp_alg,m,n,L,rmax,alpha,t,features[:,:,i])

	features = torch.FloatTensor(features)
	return features,labels,idx_train,idx_val,idx_test,memory_dataset


def load_multi_inductive(datastr, dataset_name, params_list):
	dataset_el = os.path.join(datastr,dataset_name+"_adj_el.txt")
	dataset_pl = os.path.join(datastr,dataset_name+"_adj_pl.txt")

	if dataset_name=="Amazon2M":
		train_m=62382461; train_n=1709997
		full_m=126167053; full_n=2449029
	if dataset_name=='yelp':
		train_m=7949403; train_n=537635
		full_m=13954819; full_n=716847
	if dataset_name=='reddit':
		train_m=10907170; train_n=153932
		full_m=23446803; full_n=232965
	py_agp=AGP()

	print('load')
	raw_features_train=np.load(os.path.join(datastr,dataset_name+'_train_feat.npy'))
	raw_features = np.load(os.path.join(datastr,dataset_name+'_feat.npy'))
	features_train = np.tile(raw_features_train, (len(params_list), 1, 1)).transpose((1,2,0))
	features = np.tile(raw_features, (len(params_list), 1, 1)).transpose((1,2,0))
	"""
	features_train = np.zeros((raw_features_train.shape[0], raw_features_train.shape[1], len(params_list)), raw_features_train.dtype)
	features  = np.zeros((raw_features.shape[0], raw_features.shape[1], len(params_list)), raw_features.dtype)
	print("--------------------------")
	for i in range(len(params_list)):
		features_train[:,:,i] = raw_features_train
		features[:,:,i] = raw_features
	features_train = np.ascontiguousarray(features_train)
	features = np.ascontiguousarray(features)
	"""
	del raw_features_train
	del raw_features
	gc.collect()

	for i, params in enumerate(params_list):
		print(params)
		agp_alg = params['agp_alg']
		L = params['L']
		rmax = params['rmax']
		alpha = params['alpha']
		t = params['ti']
		data_dir = os.path.join(datastr, f"{dataset_name}_{agp_alg}_{L}_{rmax}_{alpha}_{t}")
		os.makedirs(data_dir, exist_ok=True)
		train_path = os.path.join(data_dir, 'train_feat_agp.npy')
		full_path = os.path.join(data_dir, 'full_feat_agp.npy')
		full_memory_path = os.path.join(data_dir, 'full_memory.pickle')

		if os.path.exists(train_path):
			print("For train features propagation: loading...")
			features_train[:,:,i] =np.load(train_path)
		else:
			print("For train features propagation: culculating...")
			_=py_agp.agp_operation(dataset_el, dataset_pl, dataset_name+'_train',agp_alg,train_m,train_n,L,rmax,alpha,t,features_train[:,:,i])

		if os.path.exists(full_path) and os.path.exists(full_memory_path) :
			print("For full features propagation: loading...")
			features[:,:,i] =np.load(full_path)
			with open(full_memory_path, 'rb') as handle:
				memory_dataset = pickle.load(handle)
		else:
			print("For full features propagation: culculating...")
			memory_dataset=py_agp.agp_operation(dataset_el, dataset_pl, dataset_name+'_full',agp_alg,full_m,full_n,L,rmax,alpha,t,features[:,:,i] )

	features_train = torch.FloatTensor(features_train)
	features = torch.FloatTensor(features)
	data = np.load(os.path.join(datastr,dataset_name+"_labels.npz"))
	labels = data['labels']
	idx_train = data['idx_train']
	idx_val = data['idx_val']
	idx_test = data['idx_test']
	labels = torch.LongTensor(labels)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)

	return features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset


def load_inductive(datastr, dataset_name, agp_alg,alpha,t,rmax,L):
	dataset_el = os.path.join(datastr,dataset_name+"_adj_el.txt")
	dataset_pl = os.path.join(datastr,dataset_name+"_adj_pl.txt")

	if dataset_name=="Amazon2M":
		train_m=62382461; train_n=1709997
		full_m=126167053; full_n=2449029
	if dataset_name=='yelp':
		train_m=7949403; train_n=537635
		full_m=13954819; full_n=716847
	if dataset_name=='reddit':
		train_m=10907170; train_n=153932
		full_m=23446803; full_n=232965
	py_agp=AGP()

	data_dir = os.path.join(datastr, f"{dataset_name}_{agp_alg}_{L}_{rmax}_{alpha}_{t}")
	os.makedirs(data_dir, exist_ok=True)
	train_path = os.path.join(data_dir, 'train_feat_agp.npy')
	full_path = os.path.join(data_dir, 'full_feat_agp.npy')
	full_memory_path = os.path.join(data_dir, 'full_memory.pickle')

	if os.path.exists(train_path):
		print("For train features propagation: loading...")
		features_train =np.load(train_path)
	else:
		print("For train features propagation: culculating...")
		features_train=np.load(os.path.join(datastr,dataset_name+'_train_feat.npy'))
		_=py_agp.agp_operation(dataset_el, dataset_pl, dataset_name+'_train',agp_alg,train_m,train_n,L,rmax,alpha,t,features_train)

	if os.path.exists(full_path) and os.path.exists(full_memory_path)  :
		print("For full features propagation: loading...")
		features =np.load(full_path)
		with open(full_memory_path, 'rb') as handle:
			memory_dataset = pickle.load(handle)
	else:
		print("For full features propagation: culculating...")
		features =np.load(os.path.join(datastr,dataset_name+'_feat.npy'))
		memory_dataset=py_agp.agp_operation(dataset_el, dataset_pl, dataset_name+'_full',agp_alg,full_m,full_n,L,rmax,alpha,t,features)

	features_train = torch.FloatTensor(features_train)
	features = torch.FloatTensor(features)
	data = np.load(os.path.join(datastr,dataset_name+"_labels.npz"))
	labels = data['labels']
	idx_train = data['idx_train']
	idx_val = data['idx_val']
	idx_test = data['idx_test']
	labels = torch.LongTensor(labels)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)

	return features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset


def load_multi_transductive(datastr, dataset_name, params_list):

	dataset_el = os.path.join(datastr,dataset_name+"_adj_el.txt")
	dataset_pl = os.path.join(datastr,dataset_name+"_adj_pl.txt")

	if(dataset_name=="papers100M"):
		m=3339184668; n=111059956

	py_agp=AGP()
	print("Load graph and initialize! It could take a few minutes...")

	raw_features =np.load(os.path.join(datastr,dataset_name+'_feat_32.npy'))

	features = np.tile(raw_features, (len(params_list), 1, 1)).transpose((1,2,0))
	del raw_features
	gc.collect()
	features = features.astype(np.float64, copy=False)
	print("features shape", features.shape)

	for i, params in enumerate(params_list):
		print(params)
		agp_alg = params['agp_alg']
		L = params['L']
		rmax = params['rmax']
		alpha = params['alpha']
		t = params['ti']
		memory_dataset= py_agp.agp_operation(dataset_el, dataset_pl, dataset_name,agp_alg,m,n,L,rmax,alpha,t,features[:,:,i])

	features = torch.FloatTensor(features)
	data = np.load(os.path.join(datastr,dataset_name+'_labels.npz'))
	train_idx = torch.LongTensor(data['train_idx'])
	val_idx = torch.LongTensor(data['val_idx'])
	test_idx =torch.LongTensor(data['test_idx'])
	train_labels = torch.LongTensor(data['train_labels'])
	val_labels = torch.LongTensor(data['val_labels'])
	test_labels = torch.LongTensor(data['test_labels'])
	train_labels=train_labels.reshape(train_labels.size(0),1)
	val_labels=val_labels.reshape(val_labels.size(0),1)
	test_labels=test_labels.reshape(test_labels.size(0),1)
	return features,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset


def load_transductive(datastr, dataset_name, agp_alg,alpha,t,rmax,L):
	dataset_el = os.path.join(datastr,dataset_name+"_adj_el.txt")
	dataset_pl = os.path.join(datastr,dataset_name+"_adj_pl.txt")

	if(dataset_name=="papers100M"):
		m=3339184668; n=111059956

	py_agp=AGP()
	print("Load graph and initialize! It could take a few minutes...")
	time_sta = time.time()
	features =np.load(os.path.join(datastr,dataset_name+'_feat_32.npy'))
	features = features.astype(np.float64, copy=False)
	time_end = time.time()
	tim = time_end- time_sta
	print(tim)

	memory_dataset= py_agp.agp_operation(dataset_el, dataset_pl, dataset_name,agp_alg,m,n,L,rmax,alpha,t,features)
	features = torch.FloatTensor(features)
	#print(features.shape)

	data = np.load(os.path.join(datastr,dataset_name+'_labels.npz'))
	train_idx = torch.LongTensor(data['train_idx'])
	val_idx = torch.LongTensor(data['val_idx'])
	test_idx =torch.LongTensor(data['test_idx'])
	train_labels = torch.LongTensor(data['train_labels'])
	val_labels = torch.LongTensor(data['val_labels'])
	test_labels = torch.LongTensor(data['test_labels'])
	train_labels=train_labels.reshape(train_labels.size(0),1)
	val_labels=val_labels.reshape(val_labels.size(0),1)
	test_labels=test_labels.reshape(test_labels.size(0),1)
	return features,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset

class SimpleDataset(Dataset):
	def __init__(self,x,y):
		self.x=x
		self.y=y
		assert self.x.size(0)==self.y.size(0)

	def __len__(self):
		return self.x.size(0)

	def __getitem__(self,idx):
		return self.x[idx],self.y[idx]

def muticlass_f1(output, labels):
	preds = output.max(1)[1]
	preds = preds.cpu().detach().numpy()
	labels = labels.cpu().detach().numpy()
	micro = f1_score(labels, preds, average='micro')
	return micro

def mutilabel_f1(y_true, y_pred):
	y_pred[y_pred > 0] = 1
	y_pred[y_pred <= 0] = 0
	return f1_score(y_true, y_pred, average="micro")
