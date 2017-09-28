#!/bin/bash

N_ITER=20
BUFFER_SIZE=1
RUN_DIR="saved_models/TEST"
PATH_DATA_TRAIN="data/lamin_b1_train.csv"
PATH_DATA_TEST="data/lamin_b1_test.csv"

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python train_model.py \
       --n_iter ${N_ITER} \
       --buffer_size ${BUFFER_SIZE} \
       --replace_interval -1 \
       --path_data_train ${PATH_DATA_TRAIN} \
       --path_data_test ${PATH_DATA_TEST} \
       --batch_size 24 \
       --nn_module ttf_v8_nn \
       --path_run_dir ${RUN_DIR}

