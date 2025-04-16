# coding=utf-8
# Copyright (c) 2020 Tencent Inc. All Rights Reserved
# ******************************************************************************
# 程序名称:     和平精英好友召回Cluster GCN预测
# 功能描述:     以玩家是否回流为标签，玩家好友关系为图，利用Cluster-GCN模型计算P值
# 创建人名:     dryao
# 创建日期:     2020/7/9
# 版本说明:     v1.0
# 公司名称:     tencent
# 修改人名:
# 修改日期:
# 修改原因:
# ******************************************************************************

"""Main script for GCN models prediction."""

import os
import numpy as np
import tensorflow as tf
import models
import utils
import s3

# install packages
# os.system('apt-get update && apt-get install -y libmetis-dev')
# os.system('pip install metis==0.2a.4')
os.system('pip install -r ./requirements.txt')
os.system('pip install sklearn update')


tf.logging.set_verbosity(tf.logging.INFO)
# Set random seed
SEED = 1
np.random.seed(SEED)

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('save_name', './mymodel.ckpt', 'Path for saving model')
flags.DEFINE_string('dataset', 'ppi', 'Dataset string.')
flags.DEFINE_string('data_prefix', 'data/smoba_subgraphs', 'Datapath prefix.')
flags.DEFINE_string('subgraphs_prefix', 'data/', 'Datapath prefix.')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('num_clusters', 50, 'Number of clusters.')
flags.DEFINE_integer('bsize', 1, 'Number of clusters for each batch.')
flags.DEFINE_integer('num_clusters_val', 5,
                     'Number of clusters for validation.')
flags.DEFINE_integer('num_clusters_test', 1, 'Number of clusters for test.')
flags.DEFINE_integer('num_layers', 5, 'Number of GCN layers.')

flags.DEFINE_string('data_input', '', 'KP data input')
flags.DEFINE_string('data_output', '', 'KP data output')
flags.DEFINE_string('tb_log_dir', '', 'KP tensorboard log')
flags.DEFINE_string('embedding_path', '', 'embedding_path')

# flags.DEFINE_string(
#     'feature_path',
#     '/cephfs/group/ieg-iegpdata-hy-dc-dm/dryao/jdqssy/callback/test_subgraph_features/',
#     'Path of subgraph features')
# flags.DEFINE_string(
#     'edge_path',
#     "/cephfs/group/ieg-iegpdata-hy-dc-dm/dryao/jdqssy/callback/test_subgraph_edges/",
#     'Path of subgraph edges')

flags.DEFINE_float(
    'diag_lambda', 1,
    'A positive number for diagonal enhancement'
)
flags.DEFINE_bool('multilabel', False, 'Multilabel or multiclass.')
flags.DEFINE_bool('layernorm', True, 'Whether to use layer normalization.')
flags.DEFINE_bool(
    'precalc', True,
    'Whether to pre-calculate the first layer (AX preprocessing).')
flags.DEFINE_bool('validation', True,
                  'Print validation accuracy after each epoch.')


# Define model evaluation function
def evaluate(sess, model, val_features_batches, val_support_batches,
             y_val_batches, val_mask_batches, placeholders):

    """
    evaluate GCN model.
        :param sess: the Tensorflow session.
        :param model: the trained GCN model.
        :param val_features_batches: the array of validation subgraph features
        :param val_support_batches: the array of validation subgraph edges
        :param y_val_batches: the array of validation subgraph labels
        :param val_mask_batches: the array of validation subgraph masks, indicating val nodes
        :return: loss, accuracy, micro f1, macro f1, pvalues
    """
    total_pred = []
    total_lab = []
    total_loss = 0
    total_acc = 0

    num_batches = len(val_features_batches)
    val_data_len = 0
    for i in range(num_batches):
        features_b = val_features_batches[i]
        support_b = val_support_batches[i]
        y_val_b = y_val_batches[i]
        val_mask_b = val_mask_batches[i]
        # for all nodes in graph
        #print(val_mask_b, len(val_mask_b))
        num_data_b = np.sum(val_mask_b)
        # print(num_data_b)
        # number of test nodes (when 1 test cluster)
        if num_data_b == 0:
            print("empty")
        else:
            feed_dict = utils.construct_feed_dict(features_b, support_b, y_val_b,
                                                  val_mask_b, placeholders)
            outs = sess.run([model.loss, model.accuracy, model.outputs],
                            feed_dict=feed_dict)

        total_pred.append(outs[2][val_mask_b])
        total_lab.append(y_val_b[val_mask_b])
        total_loss += outs[0] * num_data_b
        total_acc += outs[1] * num_data_b
        val_data_len += num_data_b

    total_pred = np.vstack(total_pred)
    total_lab = np.vstack(total_lab)
    loss = total_loss / val_data_len
    acc = total_acc / val_data_len
    # print(total_pred)

    pvalues = np.array(total_pred[:, 1])

    micro, macro, report = utils.calc_f1(
        total_pred, total_lab, FLAGS.multilabel)
    print(report)
    return loss, acc, micro, macro, pvalues


def main(unused_argv):
    """Main function for running experiments."""

    data_input = FLAGS.data_input.split(',')
    edge_input = data_input[0]
    node_input = data_input[1]
    model_input = data_input[2]

    data_output = FLAGS.data_output
    log_dir = FLAGS.tb_log_dir


    # Load training nodes
    edge_path = s3.pull_train(edge_input, 'edge')
    feature_path = s3.pull_train(node_input, 'node')
    model_path = s3.pull_train(model_input, 'model')
    embedding_path = s3.create_local_path('/mnt/cephfs/embedding_path/')

    # Load test data
    (test_features_batches, test_support_batches, y_test_batches,
     test_mask_batches) = utils.load_test_two_folders(
         feature_path, edge_path, FLAGS.num_clusters_test, "test")

    model_func = models.GCN

    # Define placeholders
    placeholders = {
        'support':
            tf.sparse_placeholder(tf.float32),
        'features':
            tf.placeholder(tf.float32),
        'labels':
            tf.placeholder(tf.float32, shape=(
                None, y_test_batches[0].shape[1])),
        'labels_mask':
            tf.placeholder(tf.int32),
        'dropout':
            tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
            tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(
        placeholders,
        input_dim=test_features_batches[0].shape[1],
        logging=True,
        multilabel=FLAGS.multilabel,
        norm=FLAGS.layernorm,
        precalc=FLAGS.precalc,
        num_layers=FLAGS.num_layers)

    # Load model (using CPU for inference)
    with tf.device('/cpu:0'):
        sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        sess_cpu.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess_cpu, model_path + "/" + FLAGS.save_name)
        # Testing
        test_cost, test_acc, micro, macro, pvalues = evaluate(
            sess_cpu, model, test_features_batches, test_support_batches,
            y_test_batches, test_mask_batches, placeholders)
        print_str = 'Test set results: ' + 'cost= {:.5f} '.format(
            test_cost) + 'accuracy= {:.5f} '.format(
                test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
        tf.logging.info(print_str)

        # with open('{}/{}'.format(p_value_path, "pvalues"), 'w') as tmp_file:
        #     str_to_write = '\n'.join([str(x) for x in pvalues])
        #     tmp_file.write(str_to_write)

        node_ids = []
        for i in range(FLAGS.num_clusters_test):
            with open(feature_path + "/" + str(i), "r") as f:
                lines = f.readlines()
                for line in lines:
                    tmp = line.strip().split("\t")
                    if len(tmp) < 20:
                        print(i, tmp, line)
                    node_ids.append(tmp[1])

        print(len(node_ids))

        lines_to_write = []
        for (i, e) in enumerate(node_ids):
            tmp_line = e + "\t" + str(pvalues[i]) + "\n"
            lines_to_write.append(tmp_line)

        with open("{}/{}".format(embedding_path, 'embedding.txt'), "w") as f:
            f.writelines(lines_to_write)

        s3.push_data(embedding_path, FLAGS.embedding_path, verbose=1)


if __name__ == '__main__':
    tf.app.run(main)
