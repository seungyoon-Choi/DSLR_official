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
    def __init__(self, F, H, C, n_head, task, dropout):
        super(GAT, self).__init__()
        self.task = task
        self.conv1 = GATConv(F, H, heads=n_head)
        self.conv2 = GATConv(H * n_head, H, heads=n_head)
        self.classifier = nn.Linear(H * n_head, C)
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
        #finding added edges
        prob_adj = pairwise_cosine_similarity(z[buffer,:].reshape(-1,z.size(1)),z[target,:])
        prob_adj = (prob_adj+1)/2

        connected = find_connected_nodes(buffer, edge_index)
        mask = torch.ones(prob_adj.size(1)).to(device)
        for i, v in enumerate(connected):
            if v in target:
                indices = torch.nonzero(target==v).squeeze().item()
                mask[indices] = 0
        mask = mask.reshape(-1,mask.size(0))
        valid_prob_adj = torch.mul(prob_adj, mask)

        v, i = valid_prob_adj.topk(knn)
        i = torch.tensor([target[item] for item in i[0]]).to(device)
        i = i.reshape(-1, i.size(0))
        buffer_index = torch.full((i.size(1),),buffer).to(device)
        buffer_index = buffer_index.reshape(-1,buffer_index.size(0))
        topk_index = torch.cat((i, buffer_index), dim=0)
        topk_index_inv = torch.stack((topk_index[1], topk_index[0]), dim=0)
        valid_edge_index = coalesce(torch.cat((topk_index, topk_index_inv), dim=-1))
        
        delete_indices = []
        target_edge_index = extract_edges_with_node(edge_index, buffer)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for i in range(target_edge_index.size(1)):
            
            output = cos(z[target_edge_index[0][i],:].reshape(-1,z.size(1)), z[target_edge_index[1][i],:].reshape(-1,z.size(1))).item()
            if output <= 0.7:
                delete_indices.append(i)
            
        deleted_edge_index = target_edge_index[:,delete_indices]

        return valid_edge_index, deleted_edge_index