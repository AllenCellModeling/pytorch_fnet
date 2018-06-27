#!/bin/bash -x

DATASET=dna
MODEL_DIR=saved_models/${DATASET}
N_IMAGES=20
GPU_IDS=${1:-0}

NN_MODULE=tiramisu.tiramisu3d
SUFFIX=_tiramisu_uncertainty_5_v2

N_MAX_PIXELS=1500000

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
	 --nn_module ${NN_MODULE} \
	 --nn_kwargs '{"out_channels":2, "up_blocks":[3,3,3,3,3], "down_blocks": [3,3,3,3,3], "bottleneck_layers":5}'\
	 --overwrite True
done

