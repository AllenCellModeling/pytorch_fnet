#!/bin/bash -x

DATASET=${1:-dna}
BUFFER_SIZE=1
N_ITER=20
RUN_DIR="saved_models/TEST"
PATH_DATASET_CSV="data/csvs/${DATASET}/train.csv"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

rm -r ${RUN_DIR}
python train_model.py \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_CSV} \
       --buffer_size ${BUFFER_SIZE} \
       --iter_checkpoint 10 \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}

