# DSLR: Diversity Enhancement and Structure Learning for Rehearsal-based Graph Continual Learning

## Introduction
Implementation of **DSLR: Diversity Enhancement and Structure Learning for Rehearsal-based Graph Continual Learning.**  

Continual learning method with replay approach on graph structured data. The structure learning framework and experience replay approach were used to prevent catastrophic forgetting.  


## Basics
1. The main train/test code is in `train.py`
2. If you want to see the continual GNN model in PyTorch Geometric `MessagePassing` grammar, refer to `model.py`
3. If you want to see the replay buffer selection stage, refer to `replay.py` 
4. If you want to see hyperparameter settings, refer to `train.py`

## Run DSLR
<pre>
<code>
1. Cora
python train.py --dataset cora --replay CD --structure yes --classes_per_task 2 --memory_size 100

2. Amazon Computer
python train.py --dataset amazoncobuy --replay CD --structure yes --classes_per_task 2 --memory_size 200

3. OGB-arxiv
python train.py --dataset ogb_arxiv --replay CD --structure yes --classes_per_task 3 --memory_size 3000

</code>
</pre>
