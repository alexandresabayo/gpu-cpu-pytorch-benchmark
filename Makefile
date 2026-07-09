SHELL := /bin/bash
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh

.PHONY: install dataset benchmark clean

install:
	$(CONDA) && conda create -n pytorch python=3.10.12 -y
	$(CONDA) && conda activate pytorch && pip install -r requirements.txt
	cp -n config.example.yaml config.yaml

dataset:
	$(CONDA) && conda activate pytorch && python -m src.generate_datasets

benchmark:
	$(CONDA) && conda activate pytorch && python main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf datasets/.mnist_cache
	rm -f datasets/jena_climate_2009_2016.csv
	rm -f datasets/jena_climate_2009_2016.csv.zip