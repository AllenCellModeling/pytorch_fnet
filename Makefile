all : normal

normal:
	python3 driver.py

test:
	python3 driver.py --data_path data_test --no_model_save --n_epochs 1 --n_batches_per_img 2


