.PHONY: *

PYTHON_EXEC := python3.10


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install
	poetry run pre-commit install
	./mmdet_install.sh  # TODO: replace with poetry if possible.
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python


migrate_dataset:
	gdown https://drive.google.com/uc?id=1CZ-TLyo7Lli5gLrfpgxTR-CZ4MrnynVP -O dataset/35.zip
	unzip -q dataset/35.zip -d dataset


get_weights:
	gdown https://drive.google.com/uc?id=1tyWK07gAvLk5JTPAl4Ia_nUBYse2kyC- -O model/weights.pth


run_training:
	poetry run $(PYTHON_EXEC) -m src.mm_detection.train


run_inference:
	poetry run $(PYTHON_EXEC) -m src.mm_detection.inference
