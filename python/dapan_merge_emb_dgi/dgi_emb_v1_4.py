
#少量子图训练,微信、QQ分开
import os

os.system('pip install networkx')
os.system('pip install sklearn')

#os.system('rm -r /dapan/wechat/dapan_wechat_emb/*.csv')
#os.system('rm -r /dapan/qq/dapan_qq_emb/*.csv')

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg
from utils import process
import sklearn.metrics
import time

#data_path = 'data/tlbb'
#dataset = 'tlbb'

# training params
batch_size = 1
nb_epochs = 20
patience = 3
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 32
ft_size = 230
sparse = True
nonlinearity = 'prelu'  # special name to separate parameters


model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)


b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
nb_sub_graphs_qq = 2000
sample_sub_graph_qq = np.random.choice(nb_sub_graphs_qq, 100)
nb_sub_graphs_wechat = 4000
sample_sub_graph_wechat = np.random.choice(nb_sub_graphs_wechat, 200)

print(sample_sub_graph_qq)
print(sample_sub_graph_wechat)

qq_subgraph_list = []
for i in range(100):#sample_sub_graph_qq
    t = time.time()
    adj, features, node_ids = process.load_subgraph("dapan/qq/subgraph_edges_aug/" + str(i), 
    "dapan/qq/subgraph_features/" + str(i), ft_size)
    qq_subgraph_list.append((adj, features, node_ids))  
    print("load qq subgraph time:", time.time() - t)

wechat_subgraph_list = []
for i in range(200):#sample_sub_graph_wechat
    t = time.time()
    adj, features, node_ids = process.load_subgraph("dapan/wechat/subgraph_edges_aug/" + str(i), 
    "dapan/wechat/subgraph_features/" + str(i), ft_size)
    wechat_subgraph_list.append((adj, features, node_ids))    
    print("load wechat subgraph time:", time.time() - t)

# Training

for epoch in range(nb_epochs):
    avg_loss = 0
    for (adj, features, node_ids) in qq_subgraph_list:

        #print(adj.shape, features.shape)
        t = time.time()
        features, _ = process.preprocess_features(features)
        
        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        #print(ft_size)

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()

        features = torch.FloatTensor(features[np.newaxis])
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
        if torch.cuda.is_available():
            #print('Using CUDA')
            model.cuda()
            features = features.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()

        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, sp_adj if sparse else adj,
                    sparse, None, None, None)

        loss = b_xent(logits, lbl)
        #print('Loss:', loss)
        avg_loss += loss
        loss.backward()
        optimiser.step()
        
        print("process and learning time:", time.time() - t)


    for (adj, features, node_ids) in wechat_subgraph_list:

        #print(adj.shape, features.shape)
        features, _ = process.preprocess_features(features)


        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        #print(ft_size)

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()

        features = torch.FloatTensor(features[np.newaxis])
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
        if torch.cuda.is_available():
            #print('Using CUDA')
            model.cuda()
            features = features.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()

        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, sp_adj if sparse else adj,
                    sparse, None, None, None)

        loss = b_xent(logits, lbl)
        #print('Loss:', loss)
        avg_loss += loss
        loss.backward()
        optimiser.step()

    avg_loss /= ((nb_sub_graphs_qq + nb_sub_graphs_wechat)/20)
    print('Loss:', avg_loss)
    if avg_loss < best:
        best = avg_loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), '/dapan/best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break


#print('Loading {}th epoch'.format(best_t))

model.load_state_dict(torch.load('/dapan/best_dgi.pkl'))

# generate embeddings

for i in range(nb_sub_graphs_qq):

    adj, features, node_ids = process.load_subgraph("dapan/qq/subgraph_edges/" + str(i), 
    "dapan/qq/subgraph_features/" + str(i), ft_size)
    #print(adj.shape, features.shape)
    features, _ = process.preprocess_features(features)

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    if torch.cuda.is_available():
        #print('Using CUDA')
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    embeds = embeds[0].cpu().numpy()
    #print(embeds.shape)

    with open("dapan/qq/dapan_qq_emb/embedding_" + str(i) + ".csv", 'w') as f:
        emb_str_list = []
        for (i, emb) in enumerate(embeds):
            node_id = node_ids[i]
            emb_str = " ".join([str(e) for e in emb])
            emb_str_list.append(node_id + "\01" +emb_str)
        str_to_write = '\n'.join(emb_str_list)
        f.write(str_to_write)


for i in range(2000, nb_sub_graphs_wechat):
    adj, features, node_ids = process.load_subgraph("dapan/wechat/subgraph_edges/" + str(i), 
    "dapan/wechat/subgraph_features/" + str(i), ft_size)
    #print(adj.shape, features.shape)
    features, _ = process.preprocess_features(features)

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    if torch.cuda.is_available():
        #print('Using CUDA')
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    embeds = embeds[0].cpu().numpy()
    #print(embeds.shape)

    with open("dapan/wechat/dapan_wechat_emb/embedding_" + str(i) + ".csv", 'w') as f:
        emb_str_list = []
        for (ite, emb) in enumerate(embeds):
            node_id = node_ids[ite]
            emb_str = " ".join([str(e) for e in emb])
            emb_str_list.append(node_id + "\01" +emb_str)
        str_to_write = '\n'.join(emb_str_list)
        f.write(str_to_write)
    print("write graph: " + str(i))