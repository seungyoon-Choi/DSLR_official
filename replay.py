import random
import matplotlib.pyplot as plt

import numpy as np 
import torch
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import KMeans


def update_random(train_per_class, matching_index_inv, partition, task, size, replay_buffer, total_data):
    purified = {}
    for tasks in range(task+1):
        for i in partition[tasks]:
            purified[i] = [matching_index_inv[int(x)] for x in train_per_class[i]]
            proportion = min(int(len(purified[i]) / total_data * size), len(purified[i]))
            memo = random.sample(range(len(purified[i])), k = proportion)
            memory = [purified[i][idx] for idx in memo]
            memory = torch.from_numpy(np.array(memory))
            if replay_buffer == None:
                replay_buffer = memory
            else:
                replay_buffer = torch.cat((replay_buffer, memory),0)
    return replay_buffer


def coverage_max(network, edge_index, type, method, train_per_class, matching_index_inv, size, replay, total_data, partition, task, features, adj, distance, distances, distances_mean, k_knn):
    
    embeds = network.encode(features, edge_index)
    
    purified = {}
    replay_buffer = None

    for cla in partition[task]:
        purified[cla] = [matching_index_inv[int(x)] for x in train_per_class[cla]]
        distances[cla] = []
        distances_mean[cla] = [] # to log mean distance between all pair of node features

    for cla in partition[task]:
        proportion = min(int(len(train_per_class[cla]) / total_data * size), len(train_per_class[cla]))

        count = 0
        memory = []
        cover = []
        
        dist_matrix = torch.cdist(embeds[purified[cla]], embeds[purified[cla]], p=2)
        
        for idx in range(len(purified[cla])):
            distances[cla].extend(dist_matrix[idx, idx:])
        distances_mean[cla].append(torch.mean(torch.tensor(distances[cla])))
        
        while count < proportion:
            cm = []
            
            for idx in range(len(purified[cla])):
                if purified[cla][idx] in cover:
                    cm.append([train_per_class[cla][idx], -1, []])
                else:
                    dist = pow(embeds[purified[cla][idx]]-embeds[purified[cla]],2)

                    counts = np.sqrt(np.sum(dist.cpu().detach().numpy(),1))

                    cm.append([train_per_class[cla][idx], len(list(set(np.where(counts<distances_mean[cla][0].item()*distance)[0]) - set(cover))), list(set(np.where(counts<distances_mean[cla][0].item()*distance)[0]) - set(cover))])

            centrality = np.array([cm[ind][1] for ind in range(len(cm))])
            
            ind = centrality.argmax()
            memory.append(train_per_class[cla][ind])
            

            cover = list(set(cover) | set(cm[ind][2]) | {ind})

            #reset
            if len(cover) >= len(purified[cla]) * 0.8:
                cover = [matching_index_inv[int(x)].item() for x in memory]

            count += 1
        
        memory = torch.from_numpy(np.array(memory))

        replay[cla] = memory

    
    for i in range(task+1):
        for cla in partition[i]:
            proportion = min(int(len(train_per_class[cla]) / total_data * size), len(train_per_class[cla]))
            
            if replay_buffer == None:
                replay_buffer = replay[cla][:proportion]
            else:
                replay_buffer = torch.cat((replay_buffer, replay[cla][:proportion]), 0)

    return replay_buffer, replay


def update_MF(network, edge_index, type, matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, train_per_class, clustering, k):

    if type == 'embedding':
        embeds = network.encode(features, edge_index)

    purified = [matching_index_inv[int(x)] for x in train[task]]

    if clustering == 'yes':
        purified_per_class = {}

        for cla in partition[task]:
            purified_per_class[cla] = [matching_index_inv[int(x)] for x in train_per_class[cla]]
            mean_features[cla] = []
            count[cla] = []
            dist[cla] = []

            if type == 'feature':
                cluster = KMeans(n_clusters = k, random_state=1, n_init=10).fit(features[purified_per_class[cla]].cpu())
                cluster_ids_x = cluster.labels_
                for i in range(k):
                    mean_features[cla] += [torch.zeros(len(features[0])).to(device)]
                    count[cla].append([])
                    dist[cla].append([])
                for j in range(len(cluster_ids_x)):
                    mean_features[cla][cluster_ids_x[j]] += features[purified_per_class[cla][j]]
                    count[cla][cluster_ids_x[j]].append(purified_per_class[cla][j])
                for i in range(k):
                    mean_features[cla][i] /= len(count[cla][i])
            elif type == 'embedding':
                cluster = KMeans(n_clusters = k, random_state=1, n_init=10).fit(embeds[purified_per_class[cla]].cpu())
                cluster_ids_x = cluster.labels_
                for i in range(k):
                    mean_features[cla] += [torch.zeros(len(features[0])).to(device)]
                    count[cla].append([])
                    dist[cla].append([])
                for j in range(len(cluster_ids_x)):
                    mean_features[cla][cluster_ids_x[j]] += features[purified_per_class[cla][j]]
                    count[cla][cluster_ids_x[j]].append(purified_per_class[cla][j])
                embeds = embeds
                for i in range(k):
                    mean_features[cla][i] /= len(count[cla][i])

        for i in partition[task]:
            #for j in range(k):
            for j in range(len(count[i])):
                for k in range(len(count[i][j])):
                    if type == 'feature':
                        dist[i][j].append([count[i][j][k], float(sum(pow(features[count[i][j][k]]-mean_features[i][j],2)))])
                    elif type == 'embedding':
                        dist[i][j].append([count[i][j][k], float(sum(pow(features[count[i][j][k]]-mean_features[i][j],2)))])

        for i in mean_features.keys():
            for j in range(len(count[i])):
                distt = np.array([dist[i][j][ind][1] for ind in range(len(dist[i][j]))])
                proportion = min(int(len(count[i][j]) / total_data * size), len(count[i][j]))
                ind = np.argpartition(distt, proportion)[:proportion]
                memory = [dist[i][j][idx][0] for idx in ind]
                memory = torch.from_numpy(np.array(memory))
                if replay_buffer == None:
                    replay_buffer = memory
                else:
                    replay_buffer = torch.cat((replay_buffer, memory),0)

    elif clustering == 'no':
        for cla in partition[task]:
            if type == 'feature':
                mean_features[cla] = torch.zeros(len(features[0])).to(device)
            elif type == 'embedding':
                mean_features[cla] = torch.zeros(embeds.size(1)).to(device)
            count[cla] = []
            dist[cla] = []

        for i in range(len(purified)):
            if type == 'feature':
                mean_features[int(labels[purified[i]])] += features[purified[i]]
            elif type == 'embedding':
                mean_features[int(labels[purified[i]])] += embeds[purified[i]]
            count[int(labels[purified[i]])].append(purified[i])

        for i in partition[task]:
            mean_features[i] /= len(count[i])
            for j in range(len(count[i])):
                if type == 'feature':
                    dist[i].append([count[i][j], float(sum(pow(features[count[i][j]]-mean_features[i],2)))])
                elif type == 'embedding':
                    dist[i].append([count[i][j], float(sum(pow(embeds[count[i][j]]-mean_features[i],2)))]) #dist 계산
            
        for i in mean_features.keys():
            distt = np.array([dist[i][ind][1] for ind in range(len(dist[i]))])
            proportion = min(int(len(count[i]) / total_data * size), len(count[i]))
            ind = np.argpartition(distt, proportion)[:proportion]
            memory = [dist[i][idx][0] for idx in ind]
            memory = torch.from_numpy(np.array(memory))
            if replay_buffer == None:
                replay_buffer = memory
            else:
                replay_buffer = torch.cat((replay_buffer, memory),0)
    
    return replay_buffer, mean_features, count, dist

    
def update_CM(network, edge_index, type, train_per_class, matching_index_inv, size, replay_buffer, total_data, partition, task, features, adj, cm, distance):
    
    if type == 'embedding':
        embeds = network.encode(features, edge_index)

    purified = {}

    for cla in partition[task]:
        cm[cla] = []
        purified[cla] = [matching_index_inv[int(x)] for x in train_per_class[cla]]

    for cla in partition[task]:
        other_class = partition[task][:]
        other_class.remove(cla)
        other = []
        for clas in other_class:
            other += purified[clas]
        for idx in range(len(purified[cla])):
            if type == 'feature':
                dist = pow(features[purified[cla][idx]]-features[other],2)
            elif type == 'embedding':
                dist = pow(embeds[purified[cla][idx]]-embeds[other],2)
            counts = np.sum(dist.cpu().detach().numpy(),1)
            cm[cla].append([train_per_class[cla][idx], len(counts[counts<distance])])
    
    for i in cm.keys():
        centrality = np.array([cm[i][ind][1] for ind in range(len(cm[i]))])
        proportion = min(int(len(train_per_class[i]) / total_data * size), len(train_per_class[i]))
        ind = np.argpartition(centrality, -proportion)[-proportion:]
        memory = [cm[i][idx][0] for idx in ind]
        memory = torch.from_numpy(np.array(memory))
        if replay_buffer == None:
            replay_buffer = memory
        else:
            replay_buffer = torch.cat((replay_buffer, memory),0)
    
    return replay_buffer, cm


def homophily(index, adj, label):
    adj_list = adj[index].nonzero()
    degree = len(adj_list)
    count = 0
    for idx in adj_list:
        if label[index] == label[idx]:
            count += 1
    homophily = count/len(adj_list) if degree != 0 else 0

    return homophily


def minmax(input):
    min_scalar = np.min(input)
    max_scalar = np.max(input)
    boundary = max_scalar - min_scalar
    temp = np.zeros(len(input))
    min = temp + min_scalar
    z = np.divide((input-min),boundary)

    return z


def update_replay(replay_type, method, network, edge_index, matching_index, matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, train_per_class, clustering, k, distance, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree, k_knn):
    with torch.no_grad():
        if task == len(train)-1:
            pass
        else:
            if replay_type == 'random':
                replay_buffer = None
                replay_buffer = update_random(train_per_class, matching_index_inv, partition, task, size, replay_buffer, total_data)
                replay_buffer = [matching_index[int(x)] for x in replay_buffer]
            elif replay_type == 'MFf':
                if task == 0:
                    mean_features, count, dist = {}, {}, {}
                replay_buffer = None
                replay_buffer, mean_features, count, dist = update_MF(network, edge_index, 'feature', matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, train_per_class, clustering, k)
                replay_buffer = [matching_index[int(x)] for x in replay_buffer]
            elif replay_type == 'MFe':
                if task == 0:
                    mean_features, count, dist = {}, {}, {}
                replay_buffer = None
                replay_buffer, mean_features, count, dist = update_MF(network, edge_index, 'embedding', matching_index_inv, size, replay_buffer, total_data, partition, task, labels, train, features, adj, device, mean_features, count, dist, train_per_class, clustering, k)
                replay_buffer = [matching_index[int(x)] for x in replay_buffer]
            elif replay_type == 'CMf':
                if task == 0:
                    cm = {}
                replay_buffer = None
                replay_buffer, cm = update_CM(network, edge_index, 'feature', train_per_class, matching_index_inv, size, replay_buffer, total_data, partition, task, features, adj, cm, distance)
            elif replay_type == 'CMe':
                if task == 0:
                    cm = {}
                replay_buffer = None
                replay_buffer, cm = update_CM(network, edge_index, 'embedding', train_per_class, matching_index_inv, size, replay_buffer, total_data, partition, task, features, adj, cm, distance)
            elif replay_type =='CD':
                if task == 0:
                    replay, distances, distances_mean = {}, {}, {}
                replay_buffer, replay = coverage_max(network, edge_index, 'embedding', 'threshold', train_per_class, matching_index_inv, size, replay, total_data, partition, task, features, adj, distance, distances, distances_mean, k_knn)

    return replay_buffer, mean_features, count, dist, cm, replay, distances, distances_mean, homophily, degree