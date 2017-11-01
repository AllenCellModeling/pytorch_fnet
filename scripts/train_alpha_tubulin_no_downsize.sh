#!/bin/bash -x

TARGET=alpha_tubulin_no_downsize
RUN_DIR="saved_models/${TARGET}"
N_ITER=50000
BUFFER_SIZE=30
PATH_DATA_TRAIN="data/alpha_tubulin_train.csv"
PATH_DATA_TEST="data/alpha_tubulin_test.csv"
GPU_IDS=${1:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python train_model.py \
       --no_checkpoint_testing \
       --n_iter ${N_ITER} \
       --buffer_size ${BUFFER_SIZE} \
       --replace_interval -1 \
       --path_train_csv ${PATH_DATA_TRAIN} \
       --path_test_csv ${PATH_DATA_TEST} \
       --batch_size 24 \
       --nn_module ttf_v8_nn \
       --path_run_dir ${RUN_DIR} \
       --scale_xy -1 \
       --scale_z -1 \
       --gpu_ids ${GPU_IDS}
