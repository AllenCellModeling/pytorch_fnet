#!/bin/bash -x
BUFFER_SIZE=4
DATASET=${1:-dna}
N_SOURCE_POINTS=${2:-1}
N_OFFSET=${3:-1}
EXCLUDE_SCENES=${4:-None}
N_ITER=${5:-50000}
GPU_IDS=${6:-0}
PATCH_SIZE_MULT=${7:-1}
RUN_DIR=saved_models/${DATASET}/final/lag_${N_OFFSET}/excl_${EXCLUDE_SCENES}
PATH_DATASET_CSV=data/csvs/train_movie_${DATASET}.csv
CLASS_DATASET=CziMovieDataset
NN_MODULE=fnet_nn_3d_params

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

#rm -r ${RUN_DIR}
python train_iterative_model.py \
       --adaptive \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_CSV} \
       --class_dataset ${CLASS_DATASET} \
       --nn_module ${NN_MODULE} \
       --bpds_kwargs '{"augment_data_rate": 1}' \
       --fds_kwargs '{"n_source_points": '${N_SOURCE_POINTS}', "n_offset": '${N_OFFSET}', "exclude_scenes": "'${EXCLUDE_SCENES}'"}' \
       --nn_kwargs '{"in_channels": '${N_SOURCE_POINTS}'}' \
       --patch_size 16 $((32*${PATCH_SIZE_MULT})) $((32*${PATCH_SIZE_MULT})) \
       --batch_size 12 \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency 2000000 \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS} \
