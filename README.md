# DSLR: Diversity Enhancement and Structure Learning for Rehearsal-based Graph Continual Learning

## Introduction
Implementation of **DSLR: Diversity Enhancement and Structure Learning for Rehearsal-based Graph Continual Learning.**  

Continual learning method with replay approach on graph structured data. The structure learning framework and experience replay approach were used to prevent catastrophic forgetting.  


## Basics
1. The main train/test code is in `train.py`
2. If you want to see the continual GNN model in PyTorch Geometric `MessagePassing` grammar, refer to `model.py`
3. If you want to see the replay buffer selection stage, refer to `replay.py`
4. If you want to see hyperparameter settings, refer to `train.py`

## Run
<pre>
<code>
1. DSLR
python train.py --replay MFf --clustering no --structure no

2. Mean feature + Diversity + Structure learning
python train.py --replay MFe --structure yes

3. Coverage-based Diversity (dist: 0.1)
python train.py --replay C_Mf --distance 0.1 --structure no

4. Coverage-based Diversity (dist: 0.1) + Structure learning
python train.py --replay C_Mf --distance 0.1 --structure yes

5. Coverage-based Diversity (dist: 0.2)
python train.py --replay C_Mf --distance 0.2--structure no

6. Coverage-based Diversity (dist: 0.2) + Structure learning
python train.py --replay C_Mf --distance 0.2--structure yes

7. Coverage-based Diversity (dist: 0.3)
python train.py --replay C_Mf --distance 0.3--structure no

8. Coverage-based Diversity (dist: 0.3) + Structure learning
python train.py --replay C_Mf --distance 0.3--structure yes
</code>
</pre>