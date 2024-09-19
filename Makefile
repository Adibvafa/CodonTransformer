# Makefile

.PHONY: test
test:
	python -m unittest discover -s tests

.PHONY: test_with_coverage
test_with_coverage:
	coverage run -m unittest discover -s tests
