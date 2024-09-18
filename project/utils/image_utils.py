import os.path
from pathlib import Path

def create_train_test_validation_sets(
        input_dir:Path,
        output_dir:Path,
        filter,
        train_pct:float) -> None:

    set_folder_paths = dict(
        train=os.path.join(output_dir, "train"),
        val=os.path.join(output_dir, "val"),
        test=os.path.join(output_dir, "test")
    )

    for folder_path in set_folder_paths.values():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    files = input_dir.rglob(filter)
