import numpy as np
import scipy.sparse as sp
import torch
import pickle
from sklearn.metrics import f1_score,precision_score,recall_score

def load_data(path="../data/dblp/", dataset="dblp"):
    """Load citation network dataset (dblp only for now)"""
    print('Loading {} dataset...'.format(dataset))
    adj_file=open('../data/dblp/edges.pkl','rb')
    label_file=open('../data/dblp/labels.pkl','rb')
    feature_file=open('../data/dblp/node_features.pkl','rb')
    adj_list=pickle.load(adj_file)
    labels_list=pickle.load(label_file)
    features_list=pickle.load(feature_file)
    adj=adj_list[0]+adj_list[1]+adj_list[2]+adj_list[3]
    label=labels_list[0]+labels_list[1]+labels_list[2]
    features=features_list
    #adj=adj.todense()
    labels=np.array(label)
    features=np.array(features)
    # build graph
   
    idx = np.array(labels[:, 0], dtype=np.int32)
   
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(0,2400)
    idx_val = range(2400, 3200)
    idx_test = range(3200, 4000)

    features = torch.FloatTensor(features)
   # labels = torch.LongTensor(np.where(labels)[1])
   # labels = torch.LongTensor(labels[1])
    labels = torch.LongTensor(labels[:, -1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1),dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
   # print(preds)
    p = precision_score(labels, preds, average='macro')
    r = recall_score(labels, preds, average='macro')
    f1 = f1_score( labels, preds, average='macro' )
    return f1
    #print(f1, p, r)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
