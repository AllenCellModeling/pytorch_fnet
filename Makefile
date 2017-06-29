all : normal

normal:
	python3 driver.py

test:
	python3 driver.py --no_model_save --data_path data_one_file --n_epochs 1 --n_batches_per_img 100

long:
	python3 driver.py --data_path data_2 --n_epochs 500 --n_batches_per_img 50 --iter_save_model 500

test_one_img:
	python3 driver.py --data_path data/one_file --n_epochs 1 --n_batches_per_img 500 --run_name one_file
