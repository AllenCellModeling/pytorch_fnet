#!/bin/bash -x

DATASET=dna
BUFFER_SIZE=1
N_ITER=20
RUN_DIR="saved_models/TEST"
GPU_IDS=${1:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

rm -rf ${RUN_DIR}
python train_model.py \
       --n_iter ${N_ITER} \
       --class_dataset DummyChunkDataset \
       --buffer_size ${BUFFER_SIZE} \
       --iter_checkpoint 10 \
       --batch_size 24 \
       --nn_module ttf_v8_nn \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}
