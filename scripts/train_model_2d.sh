#!/bin/bash -x

DATASET=${1:-LMmed}
BUFFER_SIZE=8
N_ITER=50000
RUN_DIR="saved_models/${DATASET}"
PATH_DATASET_ALL_CSV="data/csvs/${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="data/csvs/${DATASET}/train.csv"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "data/csvs" -v
python train_model.py \
       --nn_module fnet_nn_2d \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --class_dataset TiffDataset \
       --transform_signal fnet.transforms.normalize \
       --transform_target fnet.transforms.normalize \
       --patch_size 256 256 \
       --batch_size 32 \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency 16000 \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS}

