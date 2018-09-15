#!/bin/bash -x
DATASET=${1:-dna}
N_SOURCE_POINTS=${2:-1}
SOURCE_POINTS=${3:-1}
N_OFFSET=${4:-1}
EXCLUDE_SCENES=${5:-None}
MODEL_DIR=${DATASET}/final_mse/lag_${N_OFFSET}/excl_${EXCLUDE_SCENES}
PATH_MODEL_DIR=saved_models/${MODEL_DIR}
N_IMAGES=20
GPU_IDS=${6:-0}
PATH_SAVE_DIR=results/3d/${MODEL_DIR}/model_${SOURCE_POINTS}

rm -r ${PATH_SAVE_DIR}
python predict_movie.py \
    --class_dataset CziMovieDataset \
    --path_model_dir ${PATH_MODEL_DIR} \
    --path_dataset_csv data/csvs/test_movie_${DATASET}.csv \
    --n_images ${N_IMAGES} \
    --iter_mode \
    --nn_module fnet_nn_3d_params \
    --fds_kwargs '{"n_source_points": '${N_SOURCE_POINTS}', "source_points": "'${SOURCE_POINTS}'", "n_offset": '${N_OFFSET}', "exclude_scenes": "'${EXCLUDE_SCENES}'"}' \
    --nn_kwargs '{"in_channels": '${N_SOURCE_POINTS}'}' \
    --no_signal \
    --no_prediction_unpropped \
    --path_save_dir ${PATH_SAVE_DIR} \
    --gpu_ids ${GPU_IDS} \

