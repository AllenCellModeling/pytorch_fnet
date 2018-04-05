#!/bin/bash -x

DATASET=${1:-dic_lamin_b1}
MODEL_DIR=saved_models/${DATASET}
N_IMAGES=20
GPU_IDS=${2:-0}

for TEST_OR_TRAIN in test train
do
    python predict.py \
	   --transform_signal fnet.transforms.normalize "fnet.transforms.Resizer((1, 0.5931, 0.5931))" \
	   --transform_target fnet.transforms.normalize "fnet.transforms.Resizer((1, 0.5931, 0.5931))" \
	   --path_model_dir ${MODEL_DIR} \
	   --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
	   --n_images ${N_IMAGES} \
	   --no_prediction_unpropped \
	   --path_save_dir results/3d/${DATASET}/${TEST_OR_TRAIN} \
	   --gpu_ids ${GPU_IDS}
done
