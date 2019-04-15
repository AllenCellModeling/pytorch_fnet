# Ideas from "Automating Code Quality" talk by Kyle Knapp

check:
	flake8 --extend-ignore=E501 --max-complexity 10 --exclude=tests/ignore/* fnet tests examples

pylint:
	pylint -d c0111,c0103 tests/*.py tests/data/*.py

test:
	pytest --ignore=tests/ignore tests

coverage:
	pytest --cov=fnet --ignore=tests/ignore tests

htmlcov:
	pytest --cov=fnet --cov-report=html --ignore=tests/ignore tests

prcheck: check test
