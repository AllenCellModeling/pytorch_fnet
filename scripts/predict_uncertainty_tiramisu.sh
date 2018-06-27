#!/bin/bash -x

DATASET=dna
MODEL_DIR=saved_models/${DATASET}
N_IMAGES=20
GPU_IDS=${1:-0}

SUFFIX=_tiramisu_uncertainty_fullsize_v4_dr10

N_MAX_PIXELS=999999999999

echo ${DATASET}${SUFFIX}

for TEST_OR_TRAIN in test train
do
  python predict_stitched.py \
	 --path_model_dir ${MODEL_DIR}${SUFFIX} \
	 --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
	 --n_images ${N_IMAGES} \
	 --no_prediction_unpropped \
	 --path_save_dir results/${DATASET}${SUFFIX}/${TEST_OR_TRAIN} \
	 --gpu_ids ${GPU_IDS} \
         --transform_signal 'fnet.transforms.normalize' \
         --transform_target 'fnet.transforms.normalize' \
	 --overwrite True \
	 --propper_kwargs '{"n_max_pixels": 1E100}'
done

