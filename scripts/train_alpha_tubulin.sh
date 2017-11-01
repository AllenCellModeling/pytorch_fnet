#!/bin/bash -v

TARGET=alpha_tubulin
RUN_DIR="saved_models/${TARGET}"
N_ITER=50000
BUFFER_SIZE=30
PATH_DATA_TRAIN="data/${TARGET}_train.csv"
PATH_DATA_TEST="data/${TARGET}_test.csv"
GPU_IDS=0

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python train_model.py \
       --n_iter ${N_ITER} \
       --buffer_size ${BUFFER_SIZE} \
       --replace_interval -1 \
       --path_train_csv ${PATH_DATA_TRAIN} \
       --path_test_csv ${PATH_DATA_TEST} \
       --batch_size 24 \
       --nn_module ttf_v8_nn \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}
