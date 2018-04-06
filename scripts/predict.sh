#!/bin/bash -x

DATASET=${1:-dna}
MODEL_DIR=saved_models/${DATASET}
N_IMAGES=20
GPU_IDS=${2:-0}
TRANSFORM_TARGET=${3:-fnet.transforms.normalize}

SUFFIX=${4:-}



echo ${DATASET}${SUFFIX}

for TEST_OR_TRAIN in test train
do
  python predict.py \
	 --path_model_dir ${MODEL_DIR}${SUFFIX} \
	 --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
	 --n_images ${N_IMAGES} \
	 --no_prediction_unpropped \
	 --path_save_dir results/${DATASET}${SUFFIX}/${TEST_OR_TRAIN} \
	 --gpu_ids ${GPU_IDS} 
done

