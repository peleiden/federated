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
	python ./src/models/train_federated.py

traincentral:
	python ./src/models/train_model.py

dataset:
	python ./src/data/make_dataset.py
