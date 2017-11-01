#!/bin/bash -v

RUN_DIR="saved_models/fibrillarin_scale_xy_0p15"
N_ITER=50000
BUFFER_SIZE=30
PATH_DATA_TRAIN="data/fibrillarin_train.csv"
PATH_DATA_TEST="data/fibrillarin_test.csv"
SCALE_XY=0.15
GPU_IDS=1

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
       --scale_xy ${SCALE_XY} \
       --gpu_ids ${GPU_IDS}
