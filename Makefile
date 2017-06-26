all : normal

normal:
	python3 driver.py

debug:
	python3 tmp_debug.py

test_mode:
	python3 driver.py --test_mode
