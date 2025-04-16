# coding=utf-8
# Copyright (c) 2020 Tencent Inc. All Rights Reserved
# ******************************************************************************
# 程序名称:     训练、预测脚本
# 功能描述:     训练模型的Shell命令、加载训练好的模型预测的Shell命令
# 创建人名:     dryao
# 创建日期:     2020/7/9
# 版本说明:     v1.0
# 公司名称:     tencent
# 修改人名:
# 修改日期:
# 修改原因:
# ******************************************************************************

#!/bin/bash
# training commands
python train_callback.py --dataset jdqssy --data_prefix /callback/ --nomultilabel --num_layers 5 --num_clusters 1000 --bsize 1 --layernorm --precalc True --hidden1 128 --dropout 0.0 --weight_decay 0  --early_stopping 1000 --num_clusters_val 1000 --num_clusters_test 1000 --epochs 100 --save_name callback/202012/clustergcn_model --diag_lambda 1 --statis_date 20200713

# test commands
python test_callback.py --dataset jdqssy --data_prefix /callback/ --nomultilabel --num_layers 5 --num_clusters 1000 --bsize 1 --layernorm --precalc True --hidden1 128 --dropout 0.0 --weight_decay 0  --early_stopping 1000 --num_clusters_val 1000 --num_clusters_test 1000 --epochs 100 --save_name callback/202012/clustergcn_model --diag_lambda 1 --statis_date 20200720