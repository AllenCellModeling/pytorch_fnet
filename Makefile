# Ideas from "Automating Code Quality" talk by Kyle Knapp

check:
	flake8 --extend-ignore=E501 --max-complexity 10 fnet tests/*.py

test:
	pytest --ignore tests/ignore tests

pylint:
	pylint -d c0111,c0103 tests/*.py

prcheck: check test
