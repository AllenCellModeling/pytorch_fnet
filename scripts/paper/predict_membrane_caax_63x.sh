#!/bin/bash -x

POSTFIX=${2}
NAME_MODEL=membrane_caax_63x${POSTFIX}
DATASET=membrane_caax_63x
PATH_MODEL_DIR=saved_models/${NAME_MODEL}
PATH_DATASET_TRAIN_CSV=data/csvs/${DATASET}/train.csv
N_IMAGES=20
GPU_IDS=${1:-0}

# resize factor: images are 0.086 um/px, want 0.29 um/px => factors (1, 0.29655, 0.29655)

for TEST_OR_TRAIN in test train
do
  python predict.py \
	 --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
	 --transform_signal fnet.transforms.normalize "fnet.transforms.Resizer((1, 0.29655, 0.29655))" \
	 --transform_target fnet.transforms.normalize "fnet.transforms.Resizer((1, 0.29655, 0.29655))" \
	 --path_model_dir ${PATH_MODEL_DIR} \
	 --n_images ${N_IMAGES} \
	 --no_prediction_unpropped \
	 --path_save_dir results/3d/${NAME_MODEL}/${TEST_OR_TRAIN} \
	 --gpu_ids ${GPU_IDS}
done
