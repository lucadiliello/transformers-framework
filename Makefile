# Clean check and install requirements

clean:
	# clean all temp runs
	rm -r lightning_logs/
	rm -r .pytest_cache
	rm -r outputs/

check:
	# isort and flake
	isort .
	flake8 .

env:
	pip install -r requirements/requirements.txt

test:
	for f in tests/scripts/*.sh; do bash $f; done
	rm -r /tmp/transformers_framework
