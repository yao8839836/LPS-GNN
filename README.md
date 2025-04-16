# LPS-GNN
<!--基于图分割和子图全量图神经网络(Graph Convoutional Networks, GCN)的大规模图神经网络框架。可用于游戏内丰富的应用场景，比如好友召回，未成年人识别、广告推荐等。本目录以游戏A好友召回为例，对2亿+游戏玩家组成的游戏社交关系图训练5层有监督GCN模型。以预测流失玩家的回流概率。-->
A large-scale graph neural network framework based on graph partitioning and subgraph-based full-scale Graph Convolutional Networks (GCN). It can be applied to diverse in-game scenarios such as friend recall, minor identification, and ad recommendation. Taking the friend recall scenario in Game A as an example, this directory trains a 5-layer supervised GCN model on a social relationship graph composed of over 200 million gamers to predict the return probability of churned players.

![image-20210106184901](../img/大规模GCN框架图.png)

### Prerequisites
* Python 3.7
* tensorflow-gpu>=1.12.0
* networkx==1.11
* numpy>=1.15.4
* scipy>=1.2.0
* scikit-learn>=0.19.1
* metis
* setuptools

## Models
<!--目前该大规模GCN框架已实现以下模型。-->
Here are some examples of supervised and unsupervised GNNs.
* `GCN`: The GCN class in models.py performs full-scale graph convolutional computation across multiple layers on subgraphs partitioned by the graph segmentation algorithm (implemented via Spark), generating node embeddings to predict node (player) labels. Each subgraph is processed as an input batch to iteratively update a global graph network model.
* `Deep Graph Infoxmax (DGI)`: ../dapan_merge_emb_dgi/models/dgi.py, Unsupervised node embedding learning is performed on subgraphs partitioned by the graph partitioning algorithm (implemented via Spark). Contrastive learning is leveraged to construct augmented views of the original subgraphs, enhancing representation learning through cross-view node similarity optimization.



## Running the tests
Training and test commands are in .python/chiji_callback/run.sh.
### Training supervised LPS-GNN model
```
python train_callback.py --dataset jdqssy --data_prefix /callback/ --nomultilabel --num_layers 5 --num_clusters 1000 --bsize 1 --layernorm --precalc True --hidden1 128 --dropout 0.0 --weight_decay 0  --early_stopping 1000 --num_clusters_val 1000 --num_clusters_test 1000 --epochs 100 --save_name callback/202012/clustergcn_model --diag_lambda 1 --statis_date 20200713
```
The model is iteratively trained based on the configured graph network parameters, achieving F1-score improvement on the validation set. Example output is shown below.
```
INFO:tensorflow:Epoch: 0001 training time: 18.29093 train_acc= 0.74551 train_loss= 0.43587 val_acc= 0.75029 mi F1= 0.75029 ma F1= 0.71874 val_loss= 0.43125 
INFO:tensorflow:Epoch: 0002 training time: 35.12918 train_acc= 0.75919 train_loss= 0.42347 val_acc= 0.75096 mi F1= 0.75096 ma F1= 0.71464 val_loss= 0.43033 
INFO:tensorflow:Epoch: 0003 training time: 51.94466 train_acc= 0.75386 train_loss= 0.42979 val_acc= 0.75107 mi F1= 0.75107 ma F1= 0.72261 val_loss= 0.43014 
INFO:tensorflow:Epoch: 0004 training time: 68.99974 train_acc= 0.73718 train_loss= 0.44806 val_acc= 0.75077 mi F1= 0.75077 ma F1= 0.72551 val_loss= 0.43039 
INFO:tensorflow:Epoch: 0005 training time: 85.67941 train_acc= 0.74795 train_loss= 0.43183 val_acc= 0.75078 mi F1= 0.75078 ma F1= 0.71321 val_loss= 0.43052 

              precision    recall  f1-score   support

           0     0.9442    0.9899    0.9665   3493007
           1     0.8165    0.4340    0.5667    360839

   micro avg     0.9379    0.9379    0.9379   3853846
   macro avg     0.8804    0.7120    0.7666   3853846
weighted avg     0.9323    0.9379    0.9291   3853846
```


### Predicting player churn probability
```
python test_callback.py --dataset jdqssy --data_prefix /callback/ --nomultilabel --num_layers 5 --num_clusters 1000 --bsize 1 --layernorm --precalc True --hidden1 128 --dropout 0.0 --weight_decay 0  --early_stopping 1000 --num_clusters_val 1000 --num_clusters_test 1000 --epochs 100 --save_name /callback/202012/clustergcn_model --diag_lambda 1 --statis_date 20200720
```
Example output, player p value
```
20210103^Aopenid1^Aroleid1^A-0.42435536
20210103^Aopenid2^Aroleid2^A1.2665211
20210103^Aopenid3^Aroleid3^A-0.594237
20210103^Aopenid4^Aroleid4^A-0.75172585
20210103^Aopenid5^Aroleid5^A-0.38986883
20210103^Aopenid6^Aroleid6^A-0.44822133
20210103^Aopenid7^Aroleid7^A-0.874013
20210103^Aopenid8^Aroleid8^A2.7857895
20210103^Aopenid9^Aroleid9^A-1.4937632
20210103^Aopenid10^Aroleid10^A-2.3853867
20210103^Aopenid11^Aroleid11^A3.0820272
20210103^Aopenid12^Aroleid12^A-0.4866414
20210103^Aopenid13^Aroleid13^A-1.3921325
```

## Python Code Structure

```
├─ inits.py                     // Implements various neural network parameter initialization methods  
├─ layers.py                     // Implements various neural network layers  
├─ metrics.py                    // Implements loss function metrics and accuracy calculation methods  
├─ models.py                     // Implements multi-layer GCN models and multi-layer perceptron (MLP) models  
├─ test_callback.py              // Prediction for friend recall in Game A 
├─ train_callback.py             // Training for friend recall in Game A 
├─ utils.py                      // Implements methods for reading Spark-generated subgraph files and various preprocessing utilities  
