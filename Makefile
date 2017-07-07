tmp:
	python3 train_model.py --data_path data/one_file --n_epochs 1 --n_batches_per_img 5000 \
	--run_name one_file_test_nosig_nonorm --img_transform do_nothing --nn_module nosigmoid_nn \
	--lr 0.001

tmp_lr01:
	python3 train_model.py --data_path data/one_file --n_epochs 1 --n_batches_per_img 500 \
	--run_name one_file_test_nosig_nonorm_lr01 --img_transform do_nothing --nn_module nosigmoid_nn \
	--lr 0.01

long:
	python3 train_model.py --data_path data/no_hots --n_iter 50000 --run_name no_hots_multi

test_one_img:
	python3 train_model.py --data_path data/one_file --n_iter 20 --run_name one_file_test

test_modules:
	python3 test_modules.py
