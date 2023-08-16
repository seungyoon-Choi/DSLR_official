import torch
import numpy as np 
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Amazon
import collections
import os.path as osp
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, to_undirected
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, coalesce
from torchmetrics.functional import pairwise_cosine_similarity
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
import copy



def load_data(args, dataset="cora", classes_per_task=2):
    
    print("Dataset : {} ". format(dataset))

    if dataset == 'cora' or dataset == 'citeseer':

        path="./data/"+dataset+"/" 

        idx_features_labels = np.genfromtxt("{}{}.content".format(path,dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:,1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:,-1])
        num_classes = labels.shape[1]

        idx = np.array(idx_features_labels[:,0],dtype=np.dtype(str))
        idx_map = {j: i for i,j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path,dataset), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.dtype(int)).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T>adj) - adj.multiply(adj.T>adj)
        features = normalize_features(features)
        adj = normalize_adj(adj+sp.eye(adj.shape[0]))
        
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        labels_np = labels.numpy()
        label_counter = collections.Counter(labels_np)
        selected_ids = [id for id, count in label_counter.items() if count > 200]
        index = list(range(len(labels)))

        selected_ids = sorted(selected_ids)
        handle = [item for item in index if labels[item] in selected_ids]

        device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')
        adj = adj_masking(adj, handle, device)
        features = feature_masking(features, handle)
        labels = label_masking(labels, handle)

        index = list(range(len(labels)))

        sorted_ids = sorted(selected_ids)

        for idx in index:
            labels[idx] = torch.tensor(sorted_ids.index(labels[idx]))

        selected_ids = list(range(len(selected_ids)))

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        total_classes = int(max(labels))+1
        num_classes = len(selected_ids)

        
        if dataset == 'cora':
            idx_train, idx_test = train_test_split(index, test_size = 0.3, random_state=1)
        elif dataset == 'citeseer':
            index_per_class = []
            for i in range(2):
                index_per_class.append([])
            for idx in index:
                if labels[index[idx]] == 0:
                    index_per_class[labels[index[idx]]].append(idx)
                else:
                    index_per_class[1].append(idx)
            idx_train, idx_test = None, None
            test_ratio = [0.15, 0.6]
            for i in range(len(index_per_class)):
                class_train, class_test = train_test_split(index_per_class[i], test_size=test_ratio[i], random_state=1)
                if idx_train == None:
                    idx_train = class_train
                else:
                    idx_train = idx_train + class_train
                if idx_test == None:
                    idx_test = class_test
                else:
                    idx_test = idx_test + class_test
        
        edge_index = adjacency_to_edge_index(adj, device)
        

    elif dataset == 'amazoncobuy':
        
        Data = Amazon('./data', 'Computers')

        index = list(range(len(Data.data.x)))
        
        edges = Data.data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])), shape=(num_nodes, num_nodes))

        adj = adj + adj.T.multiply(adj.T>adj) - adj.multiply(adj.T>adj)
        adj = normalize_adj(adj+sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))

        features = Data.data.x
        labels = Data.data.y

        labels_np = labels.numpy()
        label_counter = collections.Counter(labels_np)
        selected_ids = [id for id, count in label_counter.items() if count > 400]

        selected_ids = sorted(selected_ids)
        handle = [item for item in index if labels[item] in selected_ids]

        device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')
        adj = adj_masking(adj, handle, device)
        features = feature_masking(features, handle)
        labels = label_masking(labels, handle)
        index = list(range(len(labels)))

        sorted_ids = sorted(selected_ids)

        for idx in index:
            labels[idx] = torch.tensor(sorted_ids.index(labels[idx]))

        selected_ids = list(range(len(selected_ids)))

        index_per_class = []
        for i in range(len(selected_ids)):
            index_per_class.append([])
        for idx in index:
            index_per_class[labels[index[idx]]].append(idx)

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        total_classes = int(max(labels))+1
        num_classes = len(selected_ids)

        idx_train, idx_test = None, None

        test_ratio = [0.3, 0.75, 0.65, 0.3, 0.9, 0.3, 0.4, 0.75]
        for i in range(len(selected_ids)):
            class_train, class_test = train_test_split(index_per_class[i], test_size=test_ratio[i], random_state=1)
            if idx_train == None:
                idx_train = class_train
            else:
                idx_train = idx_train + class_train
            if idx_test == None:
                idx_test = class_test
            else:
                idx_test = idx_test + class_test
        
        edge_index = adjacency_to_edge_index(adj, device)

    elif dataset == 'ogb_arxiv':
        Data = PygNodePropPredDataset(name = 'ogbn-arxiv')
        index = list(range(len(Data.data.x)))
        edges = Data.data.edge_index
        num_nodes = max(max(edges[0]), max(edges[1])) + 1
        features = Data.data.x
        labels = Data.data.y.squeeze()
        labels_np = labels.numpy()
        label_counter = collections.Counter(labels_np)
        selected_ids = [id for id, count in label_counter.items() if count > 2830]
        selected_ids = sorted(selected_ids)
        handle = [item for item in index if labels[item] in selected_ids]
        device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')
        edge_index = edge_masking(edges, handle, device)
        matching_index_inv = {}
        for idx in range(len(handle)):
            matching_index_inv[int(handle[idx])] = torch.tensor(idx).to(device)
        edge_index = torch.tensor([[matching_index_inv[edge[0].item()] for edge in edge_index.t()], [matching_index_inv[edge[1].item()] for edge in edge_index.t()]], dtype=torch.int64)

        features = feature_masking(features, handle)
        labels = label_masking(labels, handle)
        index = list(range(len(labels)))
        sorted_ids = sorted(selected_ids)
        for idx in index:
            labels[idx] = torch.tensor(sorted_ids.index(labels[idx]))
        selected_ids = list(range(len(selected_ids)))
        index_per_class = []
        for i in range(len(selected_ids)):
            index_per_class.append([])
        for idx in index:
            index_per_class[labels[index[idx]]].append(idx)
        features = normalize_features(features)
        features = torch.FloatTensor(features)
        total_classes = int(max(labels))+1
        num_classes = len(selected_ids)

        idx_train, idx_test = None, None

        test_ratio = [0.3, 0.3, 0.3, 0.3, 0.35, 0.85, 0.2, 0.2, 0.8, 0.3, 0.3, 0.8, 0.4, 0.35, 0.25]
        for i in range(len(selected_ids)):
            class_train, class_test = train_test_split(index_per_class[i], test_size=test_ratio[i], random_state=1)
            if idx_train == None:
                idx_train = class_train
            else:
                idx_train = idx_train + class_train
            if idx_test == None:
                idx_test = class_test
            else:
                idx_test = idx_test + class_test
        

    train = {}
    partition = {}


    class_per_task = classes_per_task
    num_tasks = len(selected_ids) // class_per_task
    for i in range(num_tasks):
        train[i] = []
        partition[i] = []

    train_per_class = {}
    test_per_class = {}
    class_idx = {}
    for i in range(len(selected_ids)):
        train_per_class[selected_ids[i]] = []
        test_per_class[selected_ids[i]] = []
        class_idx[selected_ids[i]] = []


    for i in range(len(idx_train)):
        if int(labels[idx_train[i]]) in selected_ids:
            if selected_ids.index(labels[idx_train[i]]) // class_per_task < num_tasks:
                train[selected_ids.index(labels[idx_train[i]])//class_per_task].append(idx_train[i])
            else:
                train[selected_ids.index(labels[idx_train[i]])//class_per_task-1].append(idx_train[i])
            train_per_class[int(labels[idx_train[i]])].append(idx_train[i])
            class_idx[int(labels[idx_train[i]])].append(idx_train[i])

    for i in range(len(selected_ids)):
        if i // class_per_task < num_tasks:
            partition[i//class_per_task].append(selected_ids[i])
        else:
            partition[i//class_per_task-1].append(selected_ids[i])
    

    for j in range(len(idx_test)):
        if int(labels[idx_test[j]]) in selected_ids:
            test_per_class[int(labels[idx_test[j]])].append(idx_test[j])
            class_idx[int(labels[idx_test[j]])].append(idx_test[j])

    for i in range(num_tasks):
        train[i] = torch.LongTensor(train[i])

    return edge_index, features, labels, train, total_classes, partition, train_per_class, test_per_class, class_idx


def adj_masking(adj, handled, device):
    adj = adj[handled,:]
    adj = adj[:,handled]

    A_tilde = adj.to(device) + torch.eye(adj.size(0)).to(device)
    D_tilde_inv_sqrt = torch.diag(torch.sqrt(torch.sum(A_tilde, dim = 1)) ** -1)
    adj = torch.mm(D_tilde_inv_sqrt, torch.mm(A_tilde, D_tilde_inv_sqrt)).to(device)
    return adj
    
def edge_masking(edge_index, handled, device):
    num_nodes = edge_index.max().item()+1
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for node in handled:
        node_mask[node] = True
    mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, mask].to(device)
    self_loop_indices = torch.tensor([[node, node] for node in handled], dtype=torch.long).t().to(device)
    edge_index = torch.cat([edge_index, self_loop_indices], dim=1).to(device)
    edge_index = coalesce(edge_index)
    return edge_index

def feature_masking(features, handled):
    features = features[handled, :]
    return features

def label_masking(labels, handled):
    labels = labels[handled]
    return labels


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def accuracy_analysis(output, labels, num_class):
    preds = output.max(1)[1].type_as(labels)
    pred = []
    for i in range(num_class):
        check = preds == i
        check = check.sum()
        pred.append(check)
    return pred

def task_accuracy(task, output, labels, class_per_task):
    preds = output[:,class_per_task*task:class_per_task*(task+1)].max(1)[1].type_as(labels)
    preds = torch.add(preds, class_per_task*task)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def normalize_adj(mx): # A_hat = DAD
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx_to =  mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return mx_to

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx_to =  r_mat_inv.dot(mx) 
    return mx_to 

def encode_onehot(labels):
    classes = sorted(set(labels))
    classes_dict = {c: np.identity(len(classes))[i,:] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def adjacency_to_edge_index(adj, device):
    edge_index = adj.nonzero(as_tuple=False).t().to(device)
    return edge_index

def edge_index_to_adjacency(edge_index, dim, device):
    adj = torch.zeros((dim, dim))
    adj[edge_index[0], edge_index[1]] = 1
    return adj.to(device)


def buffer_linkpred(model_lp, data, features, train_idx, current_idx, replay_buffer, ori_edge_index, device, knn, thres, beta, replay):
    data = data.to(device)
    optimizer_lp = torch.optim.Adam(params=model_lp.parameters(), lr=0.01)

    if replay == 'C_Mf' or replay == 'MFf':
        x_dist = torch.cdist(features[replay_buffer,:], features, p=2)
    else:
        embeds = model_lp.encode(features, data.edge_index)
        x_dist = torch.cdist(embeds[replay_buffer,:], embeds, p=2)

    model_lp = train_lp(model_lp, data, train_idx, current_idx, replay_buffer, optimizer_lp, beta, device)
    z = model_lp.encode(features, data.edge_index)
    
    buffers = replay_buffer.to(device)

    added_edge_index = None
    delete_edge_index = None
    for i, buffer in enumerate(buffers):
        buffer_sim = x_dist[i,:]
        buffer_sim = torch.negative(buffer_sim)
        
        v, target = buffer_sim.topk(50)
        
        valid_edge_index, deleted_edge_index = model_lp.decode_target(z, data.edge_index, target, buffer, knn, thres)
        valid_edge_index = valid_edge_index.to(device)
        deleted_edge_index = deleted_edge_index.to(device)
        
        if valid_edge_index != None:
            if added_edge_index != None:
                added_edge_index = torch.cat((added_edge_index, valid_edge_index), dim=-1)
            else:
                added_edge_index = valid_edge_index
        if deleted_edge_index != None:
            if delete_edge_index != None:
                delete_edge_index = torch.cat((delete_edge_index, deleted_edge_index), dim=-1)
            else:
                delete_edge_index = deleted_edge_index
        
    ori_edge_index = torch.cat((ori_edge_index, added_edge_index), dim=-1)
    ori_edge_index = coalesce(ori_edge_index)
    
    rows, cols = ori_edge_index
    edge_mask = torch.ones(rows.size(0), dtype=torch.bool).to(device)
    for i in range(delete_edge_index.size(1)):
        mask = ~((rows == delete_edge_index[0,i]) & (cols == delete_edge_index[1,i]))
        edge_mask &= mask
    ori_edge_index = ori_edge_index[:, edge_mask]
    
    return model_lp, ori_edge_index


def train_lp(model_lp, data, train_idx, current_idx, replay_idx, optimizer_lp, beta, device):
    valid_pos = None
    criterion = nn.CrossEntropyLoss
    for epoch in range(1, 100):
        model_lp.train()
        neg_edge_index = negative_sampling(
                        edge_index = data.edge_index,
                        num_nodes = data.num_nodes,
                        num_neg_samples = int(data.edge_index.size(1)*0.5))
        link_logits = get_link_logits(model_lp, data.x, data.edge_index, valid_pos, neg_edge_index)
        optimizer_lp.zero_grad()
        link_labels = get_link_labels(data.edge_index, neg_edge_index, device)
        link_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

        preds = model_lp(data.x, data.edge_index)
        current_loss = criterion()(preds[current_idx], data.y[current_idx])
        if replay_idx != None:
            replay_loss = criterion()(preds[replay_idx], data.y[replay_idx])
            if beta != 0:
                label_loss = beta * current_loss + (1-beta) * replay_loss
            elif beta == 0:
                new_beta = len(replay_idx)/(len(current_idx)+len(replay_idx))
                label_loss = new_beta * current_loss + (1-new_beta) * replay_loss
        else:
            label_loss = current_loss

        loss = link_loss + label_loss
        loss.backward()
        optimizer_lp.step()
    
    return model_lp

def get_link_logits(model_lp, x, pos_edge_index, valid_pos, neg_edge_index):
    z = model_lp.encode(x, pos_edge_index) # encode
    if valid_pos == None:
        link_logits = model_lp.decode(z, pos_edge_index, neg_edge_index) # decode
    else:
        link_logits = model_lp.decode(z, valid_pos, neg_edge_index)
    return link_logits

def get_link_labels(pos_edge_index, neg_edge_index, device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def extract_edges_with_node(edge_index, node_index):
    row, col = edge_index
    mask = ((row == node_index) | (col == node_index))
    new_edge_index = edge_index[:, mask]
    return new_edge_index

def find_connected_nodes(node, edge_index):
    connected_nodes = edge_index[1, edge_index[0]==node]
    return connected_nodes
