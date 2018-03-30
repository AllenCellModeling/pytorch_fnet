#!/bin/bash -x

N_IMAGES=20
GPU_IDS=${1:-0}

for DATASET in alpha_tubulin beta_actin desmoplakin dic_lamin_b1 dic_membrane dna fibrillarin lamin_b1 membrane myosin_iib sec61_beta st6gal1 tom20 zo1
do
    echo ${DATASET}
    for TEST_OR_TRAIN in test train
    do
	python predict.py \
	       --transform_signal fnet.transforms.do_nothing "fnet.transforms.Resizer((1, 0.37241, 0.37241))" \
	       --transform_target fnet.transforms.do_nothing "fnet.transforms.Resizer((1, 0.37241, 0.37241))" \
	       --path_dataset_csv data/csvs/${DATASET}/${TEST_OR_TRAIN}.csv \
	       --n_images ${N_IMAGES} \
	       --no_signal \
	       --no_prediction_unpropped \
	       --path_save_dir results/3d_no_target_norm/${DATASET}/${TEST_OR_TRAIN} \
	       --gpu_ids ${GPU_IDS}
    done
done
