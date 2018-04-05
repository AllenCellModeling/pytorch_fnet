#!/bin/bash -x
# for all models except membrane_caax models

GPU_IDS=${1:-0}
DATASET=${2:-dna}
SUFFIX=${3}

NAME_MODEL=${DATASET}${SUFFIX}
PATH_MODEL_DIR=saved_models/${NAME_MODEL}
N_IMAGES=20

for TEST_OR_TRAIN in test train
do
    PATH_SAVE_DIR=results/3d/${NAME_MODEL}/${TEST_OR_TRAIN}
    python predict.py \
	   --path_model_dir ${PATH_MODEL_DIR} \
  	   --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
  	   --n_images ${N_IMAGES} \
  	   --no_prediction_unpropped \
  	   --path_save_dir ${PATH_SAVE_DIR} \
  	   --gpu_ids ${GPU_IDS}
done

