import os.path
from pathlib import Path

def create_train_test_validation_sets(
        input_dir:Path,
        output_dir:Path,
        filter) -> None:

    files = input_dir.rglob(filter)
    output_folders = ["train", "val", "test"]

    for output_folder in output_folders:
        full_path = os.path.join(output_dir, output_folder)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

