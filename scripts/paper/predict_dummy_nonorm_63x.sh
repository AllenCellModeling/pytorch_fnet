#!/bin/bash -x

N_IMAGES=20
GPU_IDS=${1:-0}

for DATASET in membrane_caax_63x
do
    echo ${DATASET}
    for TEST_OR_TRAIN in test train
    do
	python predict.py \
	       --transform_signal fnet.transforms.do_nothing "fnet.transforms.Resizer((1, 0.29655, 0.29655))" \
	       --transform_target fnet.transforms.do_nothing "fnet.transforms.Resizer((1, 0.29655, 0.29655))" \
	       --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
	       --n_images ${N_IMAGES} \
	       --no_signal \
	       --no_prediction_unpropped \
	       --path_save_dir results/3d_no_target_norm/${DATASET}/${TEST_OR_TRAIN} \
	       --gpu_ids ${GPU_IDS}
    done
done
