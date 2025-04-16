# coding=utf-8
# Copyright (c) 2020 Tencent Inc. All Rights Reserved
# ******************************************************************************
# 程序名称:     和平精英好友召回Cluster GCN训练
# 功能描述:     实现了不同的神经网络层
# 创建人名:     dryao
# 创建日期:     2020/6/9
# 版本说明:     v1.0
# 公司名称:     tencent
# 修改人名:
# 修改日期:
# 修改原因:
# ******************************************************************************

"""Main script for training GCN models."""


import os
os.system('pip install -r ./requirements.txt')
os.system('pip install sklearn update')

import time
import numpy as np
import tensorflow as tf
import utils
import models
import s3
from sklearn import metrics

# install packages
# os.system('apt-get update && apt-get install -y libmetis-dev')
# os.system('pip install metis==0.2a.4')

tf.logging.set_verbosity(tf.logging.INFO)
# Set random seed
SEED = 1
np.random.seed(SEED)

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('save_name', 'mymodel.ckpt', 'Path for saving model')
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

    pvalues = np.array(total_pred[:, 1])

    micro, macro, report = utils.calc_f1(
        total_pred, total_lab, FLAGS.multilabel)
    print(report)
    
    y_true = np.argmax(total_lab, axis=1)
  
    print(y_true)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, pvalues, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    
    return loss, acc, micro, macro, pvalues


def main(unused_argv):
    """Main function for running experiments."""
    # Load data

    data_input = FLAGS.data_input.split(',')
    edge_input = data_input[0]
    node_input = data_input[1]
    mask_input = data_input[2]

    data_output = FLAGS.data_output
    log_dir = FLAGS.tb_log_dir


    # Load training nodes

    edge_path = s3.pull_train(edge_input, 'edge')
    feature_path = s3.pull_train(node_input, 'node')
    mask_path = s3.pull_train(mask_input, 'mask')

    model_path = s3.create_local_path('{}/model'.format(s3.DEFAULT_LOCAL_MODEL_PATH))


    train_ids = set()
    train_id_file = '{}/train_ids'.format(mask_path)
    # train_id_file = "/cephfs/group/ieg-iegpdata-hy-dc-dm/dryao/jdqssy/callback/train_features"

    with open(train_id_file, "r") as tmp_file:
        lines = tmp_file.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            train_ids.add(tmp[1])

    # Load validation nodes
    val_ids = set()
    val_id_file = '{}/val_ids'.format(mask_path)
    # val_id_file = "/cephfs/group/ieg-iegpdata-hy-dc-dm/dryao/jdqssy/callback/val_features"

    with open(val_id_file, "r") as tmp_file:
        lines = tmp_file.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            val_ids.add(tmp[1])

    # load subgraphs and do preprocessing

    (features_batches, support_batches, y_train_batches,
     train_mask_batches) = utils.load_train_eval_two_folders(
         feature_path, edge_path, FLAGS.num_clusters, train_ids, "train")

    (val_features_batches, val_support_batches, y_val_batches,
     val_mask_batches) = utils.load_train_eval_two_folders(
         feature_path, edge_path, FLAGS.num_clusters, val_ids, "eval")

    idx_parts = list(range(FLAGS.num_clusters))

    # Some preprocessing
    model_func = models.GCN
    
    print(features_batches[0], support_batches[0], y_train_batches[0], train_mask_batches[0])

    # Define placeholders
    placeholders = {
        'support':
            tf.sparse_placeholder(tf.float32),
        'features':
            tf.placeholder(tf.float32),
        'labels':
            tf.placeholder(tf.float32, shape=(
                None, y_train_batches[0].shape[1])),
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
        input_dim=features_batches[0].shape[1],
        logging=True,
        multilabel=FLAGS.multilabel,
        norm=FLAGS.layernorm,
        precalc=FLAGS.precalc,
        num_layers=FLAGS.num_layers)

    # Initialize session
    # comment begins
    sess = tf.Session()
    tf.set_random_seed(SEED)

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    cost_val = []
    total_training_time = 0.0
    # Train model
    for epoch in range(FLAGS.epochs):
        start_time = time.time()
        np.random.shuffle(idx_parts)
        if FLAGS.bsize > 1:
            pass
        else:
            np.random.shuffle(idx_parts)
            for pid in idx_parts:
                # Use preprocessed batch data
                features_b = features_batches[pid]
                support_b = support_batches[pid]
                y_train_b = y_train_batches[pid]
                train_mask_b = train_mask_batches[pid]
                # Construct feed dictionary
                print(features_b, support_b, y_train_b, train_mask_b)
                print(len(features_b), len(support_b),
                      len(y_train_b), len(train_mask_b))
                feed_dict = utils.construct_feed_dict(features_b, support_b, y_train_b,
                                                      train_mask_b, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy],
                                feed_dict=feed_dict)

        total_training_time += time.time() - start_time
        print_str = 'Epoch: %04d ' % (epoch + 1) + 'training time: {:.5f} '.format(
            total_training_time) + 'train_acc= {:.5f} '.format(
                outs[2]) + 'train_loss= {:.5f} '.format(outs[1])

        # Validation
        if FLAGS.validation:
            with tf.device('/cpu:0'):
                cost, acc, micro, macro, _ = evaluate(sess, model, val_features_batches,
                                                      val_support_batches, y_val_batches,
                                                      val_mask_batches,
                                                      placeholders)
                cost_val.append(cost)
                print_str += 'val_acc= {:.5f} '.format(
                    acc) + 'mi F1= {:.5f} ma F1= {:.5f} '.format(
                        micro, macro) + 'val_loss= {:.5f} '.format(cost)

                tf.logging.info(print_str)

                if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
                        cost_val[-(FLAGS.early_stopping + 1):-1]):
                    tf.logging.info('Early stopping...')
                    break

    tf.logging.info('Optimization Finished!')

    # Save model

    saver.save(sess, model_path + "/" + FLAGS.save_name)
    print(model_path, data_output)
    s3.push_model(model_path, data_output, verbose=1)

if __name__ == '__main__':
    tf.app.run(main)
