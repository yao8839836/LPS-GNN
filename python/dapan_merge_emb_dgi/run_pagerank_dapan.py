import os

os.system("pip install networkx")
os.system("pip install Node2Vec")

import networkx as nx
from node2vec import Node2Vec
from math import log
import numpy as np

G = nx.DiGraph(nx.path_graph(4))
pr = nx.pagerank(G, alpha=0.9)

print(pr)

orig_edge_file = "dapan/wechat/subgraph_edges/"
aug_edge_file = "dapan/wechat/subgraph_edges_aug/"

# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)


num_subgraph = 2000

remove_edge_ratio = 0.05

for subgraph_idx in range(0, 100):

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
            node_set.add(temp[2])
            node_set.add(temp[3])
            line_prefix = "\t".join(temp[:2])
    
        node_list = list(node_set)
        for (i, ele) in enumerate(node_list):
            node_dict[ele] = str(i)

        for line in lines:
            temp = line.strip().split("\t")
            src_id = node_dict[temp[2]]
            dst_id = node_dict[temp[3]]
            intimacy = float(temp[4])
            edge_w = log(1.1 + intimacy)
            graph.add_edge(src_id, dst_id, weight = edge_w)
            edge_list.append((src_id, dst_id))

            edge_str_set.add(src_id + "," + dst_id)
            edge_str_set.add(dst_id + "," + src_id)
        
        pr = nx.pagerank(graph, alpha=0.85)

        #print(pr)

        pr_v_list = []

        for node_id in pr:
            pr_v = pr[node_id]
            pr_v_list.append(pr_v)
        
        a = np.array(pr_v_list)
        a_sort = np.sort(a)

        low_target = int(remove_edge_ratio * len(pr_v_list))

        low_threshold = a_sort[low_target]
        print(low_target, low_threshold)


        high_target = int((1 - remove_edge_ratio) * len(pr_v_list))

        high_threshold = a_sort[high_target]        

        print(high_target, high_threshold)

        for line in lines:
            temp = line.strip().split("\t")
            src_id = node_dict[temp[2]]
            dst_id = node_dict[temp[3]]

            src_pr = pr[src_id]
            dst_pr = pr[dst_id]

            if src_pr >= low_threshold or dst_pr >= low_threshold:

                removed_edge_list.append(line)

        #aug_graph_lines = to_add_edge_list + removed_edge_list
        aug_graph_lines = removed_edge_list
        
        with open(aug_edge_file + str(subgraph_idx), "w") as tmp_file_to_write:
            tmp_file_to_write.writelines(aug_graph_lines)            
        






