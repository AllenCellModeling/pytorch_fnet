NN = v5_nn
RUN_NAME = no_hots_norm_v5_50k
DATA = data/no_hots

long:
	python3 train_model.py --data_path $(DATA) --n_iter 50000 --buffer_size 15 --replace_interval -1 \
	--nn_module $(NN) \
	--run_name $(RUN_NAME)

snm :
	python train_model.py --n_iter 500000 --buffer_size 15 --replace_interval -1 \
	--data_path data/nuc_mask \
	--data_set_module nucmaskdataset \
	--model_module snm_model \
	--nn_module snm_v0_nn \
	--lr 0.0001 \
	--run_name snm_v0

test_one_file:
	python3 train_model.py --data_path data/one_file --n_iter 1000 \
	--buffer_size 1 \
	--replace_interval -1 \
	--nn_module $(NN) \
	--run_name test_one_file

test_dataset :
	python -m unittest -v tests.test_dataset
