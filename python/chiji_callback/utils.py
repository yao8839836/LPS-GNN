# coding=utf-8
# Copyright (c) 2020 Tencent Inc. All Rights Reserved
# ******************************************************************************
# 程序名称:     preprocessing functions
# 功能描述:     实现了各种预处理方法, 以及读Spark产生的子图文件方法
# 创建人名:     dryao
# 创建日期:     2020/6/9
# 版本说明:     v1.0
# 公司名称:     tencent
# 修改人名:
# 修改日期:
# 修改原因:
# ******************************************************************************

"""Collections of preprocessing functions for different graph formats."""

from math import log
import scipy.sparse as sp
import sklearn.metrics
import sklearn.preprocessing
import numpy as np


def sample_mask(idx, length):
    """Create mask."""
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sym_normalize_adj(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj


def normalize_adj(adj):
    """Normalization by D^{-1} A."""
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (np.maximum(1.0, rowsum))
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    return adj


def normalize_adj_diag_enhance(adj, diag_lambda):
    """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(m_x):
        if not sp.isspmatrix_coo(m_x):
            m_x = m_x.tocoo()
        coords = np.vstack((m_x.row, m_x.col)).transpose()
        values = m_x.data
        shape = m_x.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for (i, _) in enumerate(sparse_mx):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def calc_f1(y_pred, y_true, multilabel):
    """Caculating f1 score."""
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    report = sklearn.metrics.classification_report(y_true, y_pred, digits=4)

    return sklearn.metrics.f1_score(
        y_true, y_pred, average='micro'), sklearn.metrics.f1_score(
            y_true, y_pred, average='macro'), report


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def preprocess_multicluster(adj,
                            parts,
                            features,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1):
    """Generate the batch for multiple clusters."""

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    np.random.shuffle(parts)
    for _, s_t in enumerate(range(0, num_clusters, block_size)):
        p_t = parts[s_t]
        for pt_idx in range(s_t + 1, min(s_t + block_size, num_clusters)):
            p_t = np.concatenate((p_t, parts[pt_idx]), axis=0)
        features_batches.append(features[p_t, :])
        y_train_batches.append(y_train[p_t, :])
        support_now = adj[p_t, :][:, p_t]
        if diag_lambda == -1:
            support_batches.append(sparse_to_tuple(normalize_adj(support_now)))
        else:
            support_batches.append(
                sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
        total_nnz += support_now.count_nonzero()

        train_pt = []
        for newidx, idx in enumerate(p_t):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(p_t)))
    return (features_batches, support_batches, y_train_batches,
            train_mask_batches)


def load_train_eval_two_folders(feature_path, edge_path, num_of_subgraphs, ids, data_split="train"):
    """Load train and eval subgraphs produced by Spark."""
    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []

    for subgraph_index in range(num_of_subgraphs):

        node_ids = []

        feature_dict = {}
        split_dict = {}
        label_dict = {}

        # Load subgraph features
        with open('{}/{}'.format(feature_path, subgraph_index), "r") as tmp_file:
            lines = tmp_file.readlines()
            for (i, line) in enumerate(lines):
                temp = line.strip().split("\t")

                node_id = temp[1]

                feature = temp[2:-1]
                # subgraph feature
                node_feature = []
                for element in feature:
                    try:
                        element = float(element)
                        if element < 0:
                            element = 0.0
                        if element > 100000000.0:
                            element = 0.0                        
                        node_feature.append(element)
                    except:
                        node_feature.append(0.0)

                feature_dict[node_id] = node_feature
                # subgraph node split
                if data_split == "train":
                    if node_id in ids:
                        node_split = "train"
                    else:
                        node_split = "eval"
                else:
                    if node_id in ids:
                        node_split = "eval"
                    else:
                        node_split = "train"
                split_dict[node_id] = node_split

                node_label = temp[-1]
                label_dict[node_id] = node_label

                if data_split == "train" and node_split != "train":
                    continue
                node_ids.append(node_id)

        num2id_dict = {}
        id2num_dict = {}
        for (i, element) in enumerate(node_ids):
            num2id_dict[i] = element
            id2num_dict[element] = i

        feature_batch = []
        # node labels
        y_train_batch = []
        train_ids = []
        num_nodes = len(node_ids)

        for i in range(num_nodes):
            node_id = num2id_dict[i]

            node_feature = feature_dict[node_id]
            feature_batch.append(node_feature)

            node_label = label_dict[node_id]
            if node_label == '0' or node_label == '0.0':
                y_train_batch.append([1, 0])
            elif node_label == '1' or node_label == '1.0':
                y_train_batch.append([0, 1])
            else:
                y_train_batch.append([0, 0])

            node_split = split_dict[node_id]
            if node_split == data_split:
                train_ids.append(i)
        feature_batch = np.array(feature_batch)
        feature_batch = sklearn.preprocessing.scale(feature_batch)
        train_mask_batch = sample_mask(train_ids, num_nodes)
        y_train_batch = np.array(y_train_batch)

        # Load subgraph edges

        row = []
        col = []
        weight = []

        with open('{}/{}'.format(edge_path, subgraph_index), "r") as tmp_file:
            lines = tmp_file.readlines()
            for line in lines:
                temp = line.strip().split("\t")
                # two node ids for the edge
                src_node_id = temp[2]
                dst_node_id = temp[3]

                intimacy = float(temp[4])
                # edge weight
                edge_w = log(1.1 + intimacy)

                if src_node_id in id2num_dict and dst_node_id in id2num_dict:
                    src_node_num = id2num_dict[src_node_id]
                    dst_node_num = id2num_dict[dst_node_id]
                    # specify the sparse adjacent matrix
                    row.append(src_node_num)
                    col.append(dst_node_num)
                    weight.append(edge_w)

        subgraph_adj = sp.csr_matrix(
            (weight, (row, col)), shape=(num_nodes, num_nodes))
        norm_subgraph_adj = sym_normalize_adj(subgraph_adj)

        features_batches.append(feature_batch)
        support_batches.append(sparse_to_tuple(norm_subgraph_adj))
        y_train_batches.append(y_train_batch)
        train_mask_batches.append(train_mask_batch)

    return (features_batches, support_batches, y_train_batches, train_mask_batches)


def load_test_two_folders(feature_path, edge_path, num_of_subgraphs, data_split="test"):
    """Load test subgraphs produced by Spark."""

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []

    for subgraph_index in range(num_of_subgraphs):

        node_ids = []

        feature_dict = {}
        split_dict = {}
        label_dict = {}

        # Load subgraph features
        with open('{}/{}'.format(feature_path, subgraph_index), "r") as tmp_file:
            lines = tmp_file.readlines()
            for (i, line) in enumerate(lines):
                temp = line.strip().split("\t")

                node_id = temp[1]

                feature = temp[2:-1]
                # subgraph feature
                node_feature = []
                for element in feature:
                    try:
                        element = float(element)
                        if element < 0:
                            element = 0.0
                        if element > 100000000.0:
                            element = 0.0                          
                        node_feature.append(element)
                    except:
                        node_feature.append(0.0)

                feature_dict[node_id] = node_feature

                node_split = "test"

                split_dict[node_id] = node_split

                node_label = temp[-1]
                label_dict[node_id] = node_label

                node_ids.append(node_id)

        num2id_dict = {}
        id2num_dict = {}
        for (i, element) in enumerate(node_ids):
            num2id_dict[i] = element
            id2num_dict[element] = i

        feature_batch = []
        y_train_batch = []
        train_ids = []
        num_nodes = len(node_ids)

        for i in range(num_nodes):
            node_id = num2id_dict[i]

            node_feature = feature_dict[node_id]
            feature_batch.append(node_feature)

            node_label = label_dict[node_id]
            if node_label == '0' or node_label == '0.0':
                y_train_batch.append([1, 0])
            elif node_label == '1' or node_label == '1.0':
                y_train_batch.append([0, 1])
            else:
                y_train_batch.append([0, 0])
                
            node_split = split_dict[node_id]
            if node_split == data_split:
                train_ids.append(i)
        feature_batch = np.array(feature_batch)
        feature_batch = sklearn.preprocessing.scale(feature_batch)
        train_mask_batch = sample_mask(train_ids, num_nodes)
        y_train_batch = np.array(y_train_batch)

        # Load subgraph edges
        row = []
        col = []
        weight = []

        with open('{}/{}'.format(edge_path, subgraph_index), "r") as tmp_file:
            lines = tmp_file.readlines()
            for line in lines:
                temp = line.strip().split("\t")
                # two node ids for the edge
                src_node_id = temp[2]
                dst_node_id = temp[3]
                # edge weight
                intimacy = float(temp[4])
                edge_w = log(1.1 + intimacy)

                if src_node_id in id2num_dict and dst_node_id in id2num_dict:
                    src_node_num = id2num_dict[src_node_id]
                    dst_node_num = id2num_dict[dst_node_id]
                    # specify the sparse adjacent matrix
                    row.append(src_node_num)
                    col.append(dst_node_num)
                    weight.append(edge_w)

        subgraph_adj = sp.csr_matrix(
            (weight, (row, col)), shape=(num_nodes, num_nodes))
        norm_subgraph_adj = sym_normalize_adj(subgraph_adj)

        features_batches.append(feature_batch)
        support_batches.append(sparse_to_tuple(norm_subgraph_adj))
        y_train_batches.append(y_train_batch)
        train_mask_batches.append(train_mask_batch)

    return (features_batches, support_batches, y_train_batches, train_mask_batches)
