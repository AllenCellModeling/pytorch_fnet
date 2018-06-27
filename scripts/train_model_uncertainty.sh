#!/bin/bash -x

DATASET=dna
N_ITER=100000
BUFFER_SIZE=30
BATCH_SIZE=32
RUN_DIR="saved_models/${DATASET}_uncertainty_fullsize"
PATH_DATASET_ALL_CSV="data/csvs/${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="data/csvs/${DATASET}/train.csv"
GPU_IDS=${1}
LR=1E-4
NN_MODULE=fnet_nn_3d_uncertainty

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "data/csvs" --train_size 0.75 -v
python train_model.py \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency 500 \
       --batch_size ${BATCH_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS} \
       --nn_kwargs '{"out_channels":2}' \
       --criterion_fn "fnet.loss_functions.MSELoss_aleotoric" \
       --lr ${LR} \
       --nn_module ${NN_MODULE} \
       --transform_signal 'fnet.transforms.normalize' \
       --transform_target 'fnet.transforms.normalize' \
       --patch_size 32 64 64
