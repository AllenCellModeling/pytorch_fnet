#!/bin/bash -x
DATASET=${1:-dna}
N_SOURCE_POINTS=${2:-1}
N_OFFSET=${3:-1}
EXCLUDE_SCENES=${4:-None}
MODEL_DIR=${DATASET}/Model_${N_SOURCE_POINTS}_${N_OFFSET}_${EXCLUDE_SCENES}
PATH_MODEL_DIR=saved_models/movies/${MODEL_DIR}
N_IMAGES=20
GPU_IDS=${5:-0}
PATH_SAVE_DIR=results/3d/${MODEL_DIR}

rm -r ${PATH_SAVE_DIR}
python predict_movie.py \
    --class_dataset CziMovieDataset \
    --path_model_dir ${PATH_MODEL_DIR} \
    --path_dataset_csv data/csvs/test_movie_${DATASET}.csv \
    --n_images ${N_IMAGES} \
    --nn_module fnet_nn_3d_params \
    --fds_kwargs '{"n_source_points": '${N_SOURCE_POINTS}', "n_offset": '${N_OFFSET}'}' \
    --nn_kwargs '{"in_channels": '${N_SOURCE_POINTS}'}' \
    --no_signal \
    --no_prediction_unpropped \
    --path_save_dir ${PATH_SAVE_DIR} \
    --gpu_ids ${GPU_IDS} 

