all : normal

normal:
	python3 driver.py

tmp:
	python3 driver.py --data_path data/one_file --n_epochs 1 --n_batches_per_img 200 --resume_path saved_models/checkpoint_test_400.p --run_name checkpoint_test_600

long:
	python3 driver.py --data_path data/tubulin_nobgsub --n_epochs 1000 --n_batches_per_img 20 --iter_save_model 500 --run_name tubulin_50_img

test_one_img:
	python3 driver.py --data_path data/one_fileg --n_epochs 1 --n_batches_per_img 500 --run_name one_file_test

test_modules:
	python3 test_modules.py
