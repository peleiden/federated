lint:
	flake8 ./src

format:
	black ./src
	isort ./src

savereqs:
	pipreqs . --force 

installreqs:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt

train:
	python ./src/models/train_model.py

