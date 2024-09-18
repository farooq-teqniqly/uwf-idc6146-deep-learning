import os.path
from pathlib import Path

def create_train_test_validation_sets(
        input_dir:Path,
        output_dir:Path,
        filter) -> None:

    files = input_dir.rglob(filter)

    training_set_folder_path = os.path.join(output_dir,"train")
    validation_set_folder_path = os.path.join(output_dir, "val")
    test_set_folder_path = os.path.join(output_dir,"test")

    if not os.path.exists(training_set_folder_path):
        os.makedirs(training_set_folder_path)

    if not os.path.exists(test_set_folder_path):
        os.makedirs(test_set_folder_path)

    if not os.path.exists(validation_set_folder_path):
        os.makedirs(validation_set_folder_path)

