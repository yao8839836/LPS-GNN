# -*- coding:utf-8 -*-
import os

DEFAULT_LOCAL_TRAIN_PATH = '/data/sns/train'
DEFAULT_LOCAL_TEST_PATH = '/data/sns/test'
DEFAULT_LOCAL_MODEL_PATH = '/data/sns/model'
DEFAULT_LOCAL_DATA_PATH = '/data/sns/data'

# pull data from s3 to local
# verbose = 0: local_path为相对路径; 否则为绝对路径
def pull_train(train_path, local_path, verbose = 0):
    print('\npull train files')
    if verbose == 0:
        local_train_path = DEFAULT_LOCAL_TRAIN_PATH if len(local_path) == 0 else \
            '{}/{}'.format(DEFAULT_LOCAL_TRAIN_PATH, local_path)
        pull(train_path, local_train_path, '*')
        return local_train_path
    else:
        pull(train_path, local_path, '*')
        return local_path

def pull_test(test_path, local_path, verbose = 0):
    print('\npull test files')
    if verbose == 0:
        local_test_path = DEFAULT_LOCAL_TEST_PATH if len(local_path) == 0 else \
            '{}/{}'.format(DEFAULT_LOCAL_TEST_PATH, local_path)
        pull(test_path, local_test_path, '*')
        return local_test_path
    else:
        pull(test_path, local_path, '*')
        return local_path

def pull(remote, local, file):
    remote = remote[:-1] if remote[-1] == '/' else remote

    print('\n[1/3] The remote files {} are:'.format(remote))
    print(os.system('s3cmd ls {}/'.format(remote)))

    print('\n[2/3] Pull remote files {} from {} to {}'.format(file, remote, local))
    os.system('mkdir -p {0}; s3cmd get --recursive --force {1}/ {0}'.format(local, remote))

    print('\n[3/3] The pulled file in {} are:'.format(local))
    print(os.system('ls -l {}'.format(local)))

# push data from local to s3
# verbose = 0: local_path为相对路径; 否则为绝对路径
def push_model(local_path, remote, verbose = 0):
    print('\nupload model to s3')
    if verbose == 0:
        model_path = DEFAULT_LOCAL_MODEL_PATH if len(local_path) == 0 else \
            '{}/{}'.format(DEFAULT_LOCAL_MODEL_PATH, local_path)
        push(remote, model_path, '*')
    else:
        push(remote, local_path, '*')

def push_data(local_path, remote, verbose = 0):
    print('\nupload data to s3')
    if verbose == 0:
        data_path = DEFAULT_LOCAL_DATA_PATH if len(local_path) == 0 else \
            '{}/{}'.format(DEFAULT_LOCAL_DATA_PATH, local_path)
        push(remote, data_path, '*')
    else:
        push(remote, local_path, '*')

def push(remote, local, file):
    remote = remote[:-1] if remote[-1] == '/' else remote

    print('\n[1/3] The local files {} are:'.format(local))
    print(os.system('ls -l {}'.format(local)))

    print('\n[2/3] Push local files {} from {} to {}'.format(file, local, remote))
    os.system('s3cmd put --recursive {0}/{1} {2}/'.format(local, file, remote))

    print('\n[3/3] The pushed file in {} are:'.format(remote))
    print(os.system('s3cmd ls {}/'.format(remote)))

def create_local_path(local):
    os.system('mkdir -p {0}'.format(local))
    return local