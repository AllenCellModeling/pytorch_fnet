check:
	flake8 --extend-ignore=E501 --max-complexity 10 fnet

test:
	pytest --ignore tests/ignore tests
