import os

os.system("pip install networkx")
os.system("pip install Node2Vec")

import networkx as nx
from node2vec import Node2Vec
from math import log
import numpy as np

orig_edge_file = "/callback/train_subgraph_edges_202012/"
aug_edge_file = "/callback/train_subgraph_edges_202012_aug/"

# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)


num_subgraph = 1000

remove_edge_ratio = 0.05 

for subgraph_idx in range(0, num_subgraph):

    with open(orig_edge_file + str(subgraph_idx), "r") as tmp_file:

        graph = nx.Graph()
        # graph.add_edge("wx1", "wx2", 0.9)

        node_set = set()
        node_dict = {}
        edge_list = []
        edge_str_set = set()

        to_add_edge_list = []
        removed_edge_list = []

        line_prefix = ""
        lines = tmp_file.readlines()
        for line in lines:
            temp = line.strip().split("\t")
            node_set.add(temp[2] + "\t" + temp[3])
            node_set.add(temp[4] + "\t" + temp[5])
            line_prefix = "\t".join(temp[:2])
    
        node_list = list(node_set)
        for (i, ele) in enumerate(node_list):
            node_dict[ele] = str(i)

        for line in lines:
            temp = line.strip().split("\t")
            src_id = node_dict[temp[2] + "\t" + temp[3]]
            dst_id = node_dict[temp[4] + "\t" + temp[5]]
            intimacy = float(temp[6])
            edge_w = log(1.1 + intimacy)
            graph.add_edge(src_id, dst_id, weight = edge_w)
            edge_list.append((src_id, dst_id))

            edge_str_set.add(src_id + "," + dst_id)
            edge_str_set.add(dst_id + "," + src_id)


        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(graph, dimensions=10, walk_length=20, num_walks=10, workers=8)  # Use temp_folder for big graphs

        # Embed nodes
        # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        model = node2vec.fit(window=10, min_count=1, batch_words=32)  

        # Look for most similar nodes
        for node_i in node_list:

            t = model.wv.most_similar(node_dict[node_i], topn = 5)  
            # Output node names are always strings
            for (sim_node, _) in t:
                tmp_str_1 = sim_node + "," + node_dict[node_i]
                tmp_str_2 = node_dict[node_i] + "," + tmp_str_1
                if tmp_str_1 not in edge_str_set and tmp_str_2 not in edge_str_set:
                    to_add_edge_list.append(line_prefix + "\t" + node_i + "\t" + node_list[int(sim_node)] + "\t" + "1.0" + "\n")



        edge_sim_list = []
        edge_sim_dict = {}

        for (src_i, dst_i) in edge_list:
            sim = model.wv.similarity(src_i, dst_i)
            print(sim)
            edge_sim_list.append(sim)
            edge_id = src_i + "," + dst_i
            edge_sim_dict[edge_id] = sim

        a = np.array(edge_sim_list)
        a_sort = np.sort(a)

        target = int(remove_edge_ratio * len(edge_sim_list))

        threshold = a_sort[target]
    
        print(target, threshold)

        for line in lines:
            temp = line.strip().split("\t")
            src_id = node_dict[temp[2] + temp[3]]
            dst_id = node_dict[temp[4] + temp[5]]
            edge_id = src_id + "," + dst_id
            edge_sim = edge_sim_dict[edge_id]
            print(edge_id, edge_sim)
            if edge_sim >= threshold:
                print(edge_id, edge_sim)
                removed_edge_list.append(line)

        aug_graph_lines = to_add_edge_list + removed_edge_list

        with open(aug_edge_file + str(subgraph_idx), "w") as tmp_file_to_write:
            tmp_file_to_write.writelines(aug_graph_lines)