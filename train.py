import time 
import random 
import argparse
import os

import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional
from model import GAT
from torch_geometric.data import Data
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

from replay import *
from utils import *


def main(args):
    # meta settings
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda:0' if(torch.cuda.is_available()) else 'cpu')


    # load the data
    adj_ori, features_ori, labels_ori, train, total_classes, partition, train_per_class, test_per_class, class_idx = load_data(args, dataset = args.dataset, classes_per_task=args.classes_per_task)
    features_ori = features_ori.to(device)
    labels_ori = labels_ori.to(device)

    #Train
    replay_buffer = None
    handle = 0

    handled = []
    accs = []
    acc = {}
    acc_class = {}
    current_accs = []
    buffer = {}
    buf_homophily = []
    for i in range(args.classes_per_task * (len(train)-1)):
        buf_homophily.append([])
    test_accuracy = {}
    train_per_class_idx = []
    for i in range(6):
        test_accuracy[i] = []

    for task in range(len(train)):

        for cla in partition[task]:
            handle += 1
            for node in class_idx[cla]:
                handled.append(node)
            buffer[cla] = []
        
        acc[task] = []
        for i in range(args.classes_per_task*task, args.classes_per_task*(task+1)):
            acc_class[i] = []

        matching_index = {} #changed -> original
        matching_index_inv = {} #original -> changed
        for idx in range(len(handled)):
            matching_index[idx] = handled[idx]
            matching_index_inv[int(handled[idx])] = torch.tensor(idx)


        train_idx = []
        val_idx = []
        test_idx = []
        whole_test_idx = []
        replay_idx = []
        current_idx = []
        test_idx_per_class = []

        for i in partition[task]:
            train_idx += [matching_index_inv[int(x)] for x in train_per_class[i]]
            current_idx += [matching_index_inv[int(x)] for x in train_per_class[i]]
            train_per_class_idx.append([])
            train_per_class_idx[i] += [matching_index_inv[int(x)] for x in train_per_class[i]]
        if replay_buffer != None:
            train_idx += [matching_index_inv[int(x)] for x in replay_buffer]
            replay_idx += [matching_index_inv[int(x)] for x in replay_buffer]

        total_data = 0

        for tasks in range(task+1):
            test_idx.append([])
            val_idx.append([])
            for i in partition[tasks]:
                test_idx[tasks] += [matching_index_inv[int(x)] for x in test_per_class[i]]
                whole_test_idx += [matching_index_inv[int(x)] for x in test_per_class[i]]
                total_data += len(train_per_class[i])
                test_idx_per_class.append([])
                test_idx_per_class[i] += [matching_index_inv[int(x)] for x in test_per_class[i]]

        if task != 0:
            buf = torch.tensor([matching_index_inv[int(x)] for x in replay_buffer])
            buf_per_class = {}
            for classs in range(task*args.classes_per_task):
                buf_per_class[classs] = []
            for i in buf:
                buf_per_class[int(labels[i])].append(i)


        #update adjacency
        if task == 0:
            ori_edge_index = adjacency_to_edge_index(adj_ori, device)
        else:
            adj_ori = edge_index_to_adjacency(ori_edge_index, adj_ori.size(0), device)
        adj = adj_masking(adj_ori, handled, device)
        features = feature_masking(features_ori, handled)
        labels = label_masking(labels_ori, handled)
        
        
        if task == 0:
            F = features.size(1) # num_of_features
            H = args.hidden # hidden nodes
            C = args.classes_per_task
            if args.model == "GCN":
                model_lp = GCN(F, H, C, task, adj, args.dropout).to(device)
            elif args.model == "GAT":
                model_lp = GAT(F, H, C, args.n_heads, task, adj, args.dropout).to(device)
        elif task != 0:
            model_lp.load_state_dict(torch.load('checkpoints/%d.pt' %(task-1)))
            if args.model == "GCN":
                weight_expand = torch.rand(C,H).to(device)
                bias_expand = torch.rand(C).to(device)
                new_weight = torch.cat((model_lp.classifier.weight, weight_expand),0)
                new_bias = torch.cat((model_lp.classifier.bias, bias_expand),0)
                model_lp.classifier.weight = nn.Parameter(new_weight)
                model_lp.classifier.bias = nn.Parameter(new_bias)
            elif args.model == "GAT":
                weight_expand = torch.rand(C,H*args.n_heads).to(device)
                bias_expand = torch.rand(C).to(device)
                new_weight = torch.cat((model_lp.classifier.weight, weight_expand),0)
                new_bias = torch.cat((model_lp.classifier.bias, bias_expand),0)
                model_lp.classifier.weight = nn.Parameter(new_weight)
                model_lp.classifier.bias = nn.Parameter(new_bias)

            model_lp.task = task
            model_lp.ori_adj = adj


        edge_index = adjacency_to_edge_index(adj, device)
        data = Data(x=features, edge_index = edge_index, y = labels)
        data.num_nodes = adj.size(0)
        data.train_mask = data.val_mask = data.test_mask = None


        
                
        ##Structure learning for replay buffer
        if task != 0 and args.structure == 'yes':
            
            model_lp, ori_edge_index = buffer_linkpred(model_lp, data, features, adj, train_idx, current_idx, replay_idx, 0.1, matching_index_inv, matching_index, replay_buffer, ori_edge_index, device, args.k_knn, args.threshold, args.beta, args.replay)
            
            adj_ori = edge_index_to_adjacency(ori_edge_index, adj_ori.size(0), device)
            adj = adj_masking(adj_ori, handled, device)
            edge_index = adjacency_to_edge_index(adj, device)
            data.edge_index = edge_index
            model_lp.ori_adj = adj


        network = model_lp
        optimizer = optim.Adam(network.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        

        for epoch in range(args.epochs):        
        
            t = time.time()
            network.train()

            preds = network(features, edge_index)

            if task == 0:
                train_loss = criterion(preds[train_idx], labels[train_idx]) 
            elif task != 0:
                current_loss = criterion(preds[current_idx], labels[current_idx])
                replay_loss = criterion(torch.index_select(preds.to(device), 0, torch.tensor(replay_idx).to(device)), torch.index_select(labels.to(device), 0, torch.tensor(replay_idx).to(device)))

                if args.beta == 0:
                    beta = len(replay_idx)/(len(current_idx)+len(replay_idx))
                else:
                    beta = args.beta

                if len(replay_idx) == 0:
                    train_loss = current_loss
                else:
                    train_loss = beta * current_loss + (1-beta) * replay_loss

            train_acc = accuracy(preds[train_idx], labels[train_idx])
            
            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()

            if (epoch+1) % 100 == 0:
                print('[%d/%d] train loss : %.4f | train acc %.2f%% | time %.3fs'
                %(epoch+1, args.epochs, train_loss.item(), train_acc.item() * 100, time.time() - t))


        # update replay buffer
        if task == 0:
            mean_features = count = dist = cm = replay = distances = distances_mean = homophily = degree = None
        replay_buffer, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree = update_replay(args.replay, args.method, network, edge_index, matching_index, matching_index_inv, args.memory_size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, train_per_class, args.clustering, args.k, args.distance, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree, args.k_knn)
        
        
        # Test
        with torch.no_grad():
            network.eval()
            preds = network(features, edge_index)
            
            for tasks in range(task+1):
                if args.incremental == 'class':
                    test_acc = accuracy(preds[test_idx[tasks]], labels[test_idx[tasks]])
                    acc[tasks].append(test_acc)
                    if task == len(train)-1:
                        current_accs.append(test_acc)                        

                elif args.incremental == 'task':
                    test_acc = task_accuracy(tasks, preds[test_idx[tasks]], labels[test_idx[tasks]], class_per_task = args.classes_per_task)
                    acc[tasks].append(test_acc)
                    if task == len(train)-1:
                        current_accs.append(test_acc)
                
                print('%d. Test Accuracy for task %s : %.2f'%(task+1, str(tasks+1), test_acc * 100))

                for classs in range(args.classes_per_task*tasks, args.classes_per_task*(tasks+1)):
                    test_acc_per_class = accuracy(preds[test_idx_per_class[classs]], labels[test_idx_per_class[classs]])
                    acc_analysis = accuracy_analysis(preds[test_idx_per_class[classs]], labels[test_idx_per_class[classs]], (task+1)*args.classes_per_task)
                    acc_analysis = [i.item() for i in acc_analysis]
                    acc_class[classs].append(test_acc_per_class)
                    print('%d. Test Accuracy for class %s : %.2f'%(classs, str(classs), test_acc_per_class * 100))
                    print('%d. Test Prediction for class %s : %s' %(classs, str(classs), acc_analysis))

            accs.append(test_acc)
            

        torch.save(network.state_dict(), 'checkpoints/%d.pt' %(task))



    if args.forget == 'task':       
        print('Average Performance : %.2f'%(average_performance(acc)))
        print('Average Performance for last task : %.2f'%(current_performance(acc)))
        print('Forgetting Performance : %.2f'%(forget_performance(acc, args.classes_per_task, args.forget)))

    elif args.forget == 'class':
        print('Average Performance : %.2f'%(average_performance(acc_class)))
        print('Average Performance for last task : %.2f'%(current_performance(acc_class)))
        print('Forgetting Performance : %.2f'%(forget_performance(acc_class, args.classes_per_task, args.forget)))
        
    

    

def average_performance(acc):
    sum = 0
    for i in range(len(acc)):
        sum += acc[i][0]
    return round((sum / len(acc)).item() * 100, 2)

def current_performance(acc):
    sum = 0
    for i in range(len(acc)):
        sum += acc[i][-1]
    return round((sum / len(acc)).item() * 100, 2)

def forget_performance(acc, classes_per_task, forget):
    sum = 0
    #count = 0
    for task in acc.keys():
        sum += (acc[task][0]-acc[task][-1])
    if forget == 'task':
        return round((sum / (len(acc.keys())-1)).item() * 100, 2)
    elif forget == 'class':
        return round((sum / (len(acc.keys())-classes_per_task)).item() * 100, 2)
    


if __name__  == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes_per_task', type=int, default=2, help='classes per task')
    parser.add_argument('--replay', type=str, default='CD', choices=['random', 'MFf', 'MFe', 'CMf', 'CMe', 'CD'], help='replay method')
    parser.add_argument('--memory_size', type=int, default = 100, help='replay buffer size')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--dataset', type=str, default='citeseer', choices=['amazoncobuy','cora','citeseer', 'ogb_arxiv'], help='Dataset to train.')
    parser.add_argument('--model', type=str, default='GAT', choices=['GCN','GAT'], help='Model to train.')
    parser.add_argument('--clustering', type=str, default='no', choices=['yes', 'no'], help='diversity')
    parser.add_argument('--k', type=int, default=4, help = 'num of clusters of each class')
    parser.add_argument('--seed_shuffle', type=int, default=4, help = 'task sequence shuffle')
    parser.add_argument('--incremental', type=str, default='class', choices=['class', 'task'], help='incremental type')
    parser.add_argument('--distance', type=float, default=0.2, help='distance threshold in CM')
    parser.add_argument('--structure', type=str, default='yes', choices=['yes', 'no'], help='supervised structure learning for buffer')
    parser.add_argument('--method', type=str, default='threshold', choices=['threshold', 'knn'], help='select criteria')
    parser.add_argument('--k_knn', type=int, default=5, help='knn for replay buffer')
    parser.add_argument('--virtual_node', type=str, default='no', choices=['yes', 'no'], help='presence or absence of virtual node')
    parser.add_argument('--beta', type=float, default=0.1, help='weight in loss function')
    parser.add_argument('--threshold', type=float, default=0.99, help='Link prediction threshold')
    parser.add_argument('--forget', type=str, default='class', choices=['task', 'class'], help='Forgetting Calculation Criteria')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)