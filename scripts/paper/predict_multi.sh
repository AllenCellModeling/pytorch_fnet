#!/bin/bash -x

DATASET=tom20
MODEL_DIR="saved_models/tom20 saved_models/lamin_b1 saved_models/sec61_beta"
N_IMAGES=3
GPU_IDS=${1:-0}

TEST_OR_TRAIN=test
python predict.py \
       --path_model_dir ${MODEL_DIR} \
       --path_dataset_csv data/csvs/${DATASET}/test.csv \
       --n_images ${N_IMAGES} \
       --no_prediction_unpropped \
       --path_save_dir results/${DATASET}_test/multi \
       --gpu_ids ${GPU_IDS}

