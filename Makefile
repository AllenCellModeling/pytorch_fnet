check:
	flake8 --ignore=E501 --max-complexity 10 fnet/cli

test:
	pytest --ignore tests/ignore tests
