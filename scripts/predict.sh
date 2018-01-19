#!/bin/bash -x

DATASET=${1:-dna}
PATH_DATASET_CSV="data/${DATASET}/test.csv"
MODEL_DIR="saved_models/${DATASET}"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python predict.py \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv ${PATH_DATASET_CSV} \
       --n_images 16 \
       --gpu_ids ${GPU_IDS}

