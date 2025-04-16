#少量子图训练,微信、QQ合并

import os
import argparse
from utils import s3

os.system('pip install networkx')
os.system('pip install sklearn')



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


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input', nargs='?', default="../data", help='inputs of the component.')
    parser.add_argument('--data_output', nargs='?', default="../saved_model", help='Model output address in KP.')
    parser.add_argument("--embedding_path", type=str, default="../emb", help="")
    parser.add_argument("--tb_log_dir", type=str, default="../logs", help="tensorboard save path")
    parser.add_argument("--epochs", type=int, default=20, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--hid_units", type=int, default=32, help="embedding size")
    parser.add_argument("--ft_size", type=int, default=230, help="feature size")
    parser.add_argument("--nb_sub_graphs", type=int, default=100, help="number of subgraphs")
    parser.add_argument("--sample_ratio", type=float, default=0.05, help="propotion of subgraphs for training")

    args = parser.parse_args()

    return args


def run():
    args = arg_parser()

    data_input = args.data_input.split(',')
    edge_input = data_input[0]
    node_input = data_input[1]

    data_output = args.data_output
    embedding_path = args.embedding_path
    epochs = args.epochs
    batch_size = args.batch_size
    log_dir = args.tb_log_dir


    # training params
    batch_size = args.batch_size
    nb_epochs = args.epochs
    patience = 3
    lr = args.lr
    l2_coef = 0.0
    drop_prob = args.dropout
    hid_units = args.hid_units
    ft_size = args.ft_size
    sparse = True
    nonlinearity = 'prelu'  # special name to separate parameters

    model = DGI(ft_size, hid_units, nonlinearity)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    edge_path = s3.pull_train(edge_input, 'edge')
    feature_path = s3.pull_train(node_input, 'node')
    model_path = s3.create_local_path('{}/merge/model/'.format(s3.DEFAULT_LOCAL_MODEL_PATH))
    emb_path = s3.create_local_path('/mnt/cephfs/emb/')


    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    nb_sub_graphs = args.nb_sub_graphs
    sample_sub_graph = np.random.choice(nb_sub_graphs, int(args.sample_ratio * nb_sub_graphs))

    print(sample_sub_graph)



    subgraph_list = []
    for i in sample_sub_graph:
        t = time.time()
        adj, features, node_ids = \
            process.load_subgraph('{}/{}'.format(edge_path, i), '{}/{}'.format(feature_path, i), ft_size)
        subgraph_list.append((adj, features, node_ids))
        print("load merge subgraph time:", time.time() - t)

    # Training

    for epoch in range(nb_epochs):
        avg_loss = 0
        for (adj, features, node_ids) in subgraph_list:

            #print(adj.shape, features.shape)
            t = time.time()
            features, _ = process.preprocess_features(features)

            nb_nodes = features.shape[0]
            ft_size = features.shape[1]
            # print(nb_nodes, ft_size)

            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            # print(adj.shape)

            if sparse:
                sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
                # print(sp_adj.shape)
            else:
                adj = (adj + sp.eye(adj.shape[0])).todense()

            features = torch.FloatTensor(features[np.newaxis])
            if not sparse:
                adj = torch.FloatTensor(adj[np.newaxis])
            if torch.cuda.is_available():
                print('Using CUDA')
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


        avg_loss /= (nb_sub_graphs/20)
        print('Loss:', avg_loss)
        if avg_loss < best:
            best = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), '{}/{}'.format(model_path, 'best_dgi.pkl'))
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break


    print('Loading {}th epoch'.format(best_t))

    s3.push_model(model_path, data_output, verbose=1)
    model.load_state_dict(torch.load('{}/{}'.format(model_path, 'best_dgi.pkl')))

    # generate embeddings

    for i in range(nb_sub_graphs):
        adj, features, node_ids = \
            process.load_subgraph('{}/{}'.format(edge_path, i), '{}/{}'.format(feature_path, i), ft_size)
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

        with open(emb_path + "embedding_" + str(i) + ".csv", 'w') as f:
            emb_str_list = []
            for (i, emb) in enumerate(embeds):
                node_id = node_ids[i]
                emb_str = " ".join([str(e) for e in emb])
                emb_str_list.append(node_id + "\t" + emb_str)
            str_to_write = '\n'.join(emb_str_list)
            f.write(str_to_write)

    s3.push_data(emb_path, embedding_path, verbose=1)


if __name__ == "__main__":
    run()