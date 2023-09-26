.PHONY: *

PYTHON_EXEC := python3.10
DROPBOX_DATASET := .dropbox_dataset

CLEARML_PROJECT_NAME := obj_det_on_synth_data
CLEARML_DATASET_NAME := synth_signs_dataset


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install
	poetry run pre-commit install
	./mmdet_install.sh  # TODO: replace with poetry if possible.
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python


#migrate_dataset:
#	# Migrate dataset to ClearML datasets.
#	rm -R $(DROPBOX_DATASET) || true
#	mkdir $(DROPBOX_DATASET)
#	wget "https://www.dropbox.com/scl/fi/nrn0y41dsfwqsrssav2eo/Classification_data.zip?rlkey=oieytodt749yzyippc6384tge&dl=0" -O $(DROPBOX_DATASET)/dataset.zip
#	unzip -q $(DROPBOX_DATASET)/dataset.zip -d $(DROPBOX_DATASET)
#	rm $(DROPBOX_DATASET)/dataset.zip
#	find $(DROPBOX_DATASET) -type f -name '.DS_Store' -delete
#	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
#	clearml-data add --files $(DROPBOX_DATASET)
#	clearml-data close --verbose
#	rm -R $(DROPBOX_DATASET)


run_training:
	poetry run $(PYTHON_EXEC) -m src.mm_detection.train
