#!/bin/bash -x
DATASET=${1:-dna}
N_SOURCE_POINTS=${2:-1}
N_OFFSET=${3:-1}
EXCLUDE_SCENES=${4:-None}
MODEL_DIR=${DATASET}/iter_unc/lag_${N_OFFSET}/excl_${EXCLUDE_SCENES}
PATH_MODEL_DIR=saved_models/${MODEL_DIR}
N_IMAGES=20
GPU_IDS=${5:-0}
PATH_SAVE_DIR=results/3d/${MODEL_DIR}/model_8_5_2_1
#N_OUT_CHANNELS=$((2*${N_TARGET_POINTS}))
N_OUT_CHANNELS=2

rm -r ${PATH_SAVE_DIR}
python predict_movie_uncertainty.py \
    --class_dataset CziMovieDataset \
    --path_model_dir ${PATH_MODEL_DIR} \
    --path_dataset_csv data/csvs/test_movie_${DATASET}.csv \
    --iter_mode \
    --n_images ${N_IMAGES} \
    --nn_module fnet_nn_3d_uncertainty \
    --fds_kwargs '{"n_source_points": '${N_SOURCE_POINTS}', "n_offset": '${N_OFFSET}', "exclude_scenes": "'${EXCLUDE_SCENES}'"}' \
    --nn_kwargs '{"in_channels": '${N_SOURCE_POINTS}', "out_channels": '${N_OUT_CHANNELS}'}' \
    --no_signal \
    --no_prediction_unpropped \
    --path_save_dir ${PATH_SAVE_DIR} \
    --gpu_ids ${GPU_IDS} 

