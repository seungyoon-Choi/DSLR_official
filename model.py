import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
from torch_geometric.utils import negative_sampling, coalesce
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity
from utils import *

device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')

class GAT(nn.Module):
    def __init__(self, F, H, C, n_head, task, adj, dropout):
        super(GAT, self).__init__()
        self.task = task
        self.conv1 = GATConv(F, H, heads=n_head)
        self.conv2 = GATConv(H * n_head, H, heads=n_head)
        self.classifier = nn.Linear(H * n_head, C)
        self.ori_adj = adj
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.classifier(x)
        return x
          
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        #x = x.relu()
        return x
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        logits = cos(z[edge_index[0]], z[edge_index[1]])
        logits = (logits+1)/2
        return logits

    def decode_target(self, z, edge_index, target, buffer, knn, thres):
        prob_adj = pairwise_cosine_similarity(z, z)
        prob_adj = (prob_adj+1)/2

        yet_connected = torch.where(self.ori_adj==0,1,0).to(device)

        valid = torch.zeros(prob_adj.size(0), prob_adj.size(1)).to(device)
        valid[buffer, target] = 1
        valid = torch.mul(yet_connected, valid)
        valid_prob_adj = torch.mul(valid, prob_adj)

        if knn > 0:
            v, i = valid_prob_adj.flatten().topk(knn)
            topk_index = torch.tensor(np.array(np.unravel_index(i.cpu().numpy(), valid_prob_adj.shape)).T).to(device).T
            topk_index_inv = torch.stack((topk_index[1], topk_index[0]), dim=0)
            valid_edge_index = coalesce(torch.cat((topk_index, topk_index_inv), dim=-1))
        elif knn == 0:
            valid_edge_index = (valid_prob_adj>thres).nonzero(as_tuple=False).t()
            valid_edge_index = coalesce(torch.cat((valid_edge_index, torch.stack((valid_edge_index[1], valid_edge_index[0]), dim=0)), dim=-1))

        delete_indices = []
        target_edge_index = extract_edges_with_node(edge_index, buffer)
        for i in range(target_edge_index.size(1)):
            if prob_adj[target_edge_index[0][i], target_edge_index[1][i]] <= 0.7:
                delete_indices.append(i)
        deleted_edge_index = target_edge_index[:,delete_indices]

        return valid_edge_index, deleted_edge_index