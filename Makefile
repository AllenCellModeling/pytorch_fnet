NN = v5_nn
RUN_NAME = ttf_memb_no_hots_00
DATA = data/dataset_saves/ttf_memb_no_hots.ds

long:
	python train_model.py --n_iter 50000 --buffer_size 15 --replace_interval -1 \
	--data_path $(DATA) \
	--batch_size 64 \
	--nn_module $(NN) \
	--run_name $(RUN_NAME)

snm :
	python train_model.py --n_iter 20 --buffer_size 1 --replace_interval -1 \
	--data_path data/nuc_mask \
	--data_set_module nucmaskdataset \
	--model_module snm_model \
	--nn_module snm_v0_nn \
	--lr 0.0001 \
	--gpu_ids 0 \
	--run_name snm_v0_tmp

long_nopad :
	python train_model_2.py --n_iter 50000 --buffer_size 1 --replace_interval -1 \
	--data_path $(DATA) \
	--lr 0.001 \
	--batch_size 6 \
	--nn_module ttf_v6_nn \
	--run_name ttf_dna_v6_no_hots

gen_dataset :
	python gen_dataset.py \
	--project ttf \
	--chan memb \
	--path_source data/no_hots

test_dataset :
	python -m unittest -v tests.test_dataset

test_whole :
	python -m unittest -v tests.test_wholeimgdataprovider
