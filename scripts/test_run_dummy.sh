#!/bin/bash -x

DATASET=dummy_chunk
BUFFER_SIZE=1
N_ITER=20
RUN_DIR="saved_models/TEST"
GPU_IDS=${1:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

rm -rf ${RUN_DIR}
python train_model.py \
       --n_iter ${N_ITER} \
       --buffer_size ${BUFFER_SIZE} \
       --iter_checkpoint 10 \
       --batch_size 24 \
       --nn_module ttf_v8_nn \
       --path_run_dir ${RUN_DIR} \
       --dataset ${DATASET} \
       --gpu_ids 0
