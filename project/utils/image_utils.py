import os.path
import random
import shutil
from pathlib import Path

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
VAL_FOLDER = "val"
SHUFFLE_ROUNDS = 3

def create_train_test_validation_sets(
        input_dir: Path,
        output_dir: Path,
        file_filter="*.jpeg",
        train_percentage: float=0.7,
        test_percentage: float=0.2) -> None:
    output_folders = [_create_output_folder(output_dir, folder_name)
                      for folder_name in [TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER]]

    image_class_folders = os.listdir(input_dir)

    for image_class_folder in image_class_folders:
        source_folder_path = os.path.join(input_dir, image_class_folder)
        files = _get_files_from_folder(source_folder_path)
        random.shuffle(files)

        train_count = int(len(files) * train_percentage)
        test_count = int(len(files) * test_percentage)

        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        validation_files = files[train_count + test_count:]

        train_folder = _create_image_class_output_folder(
            output_folders[0],
            image_class_folder)

        _copy_files(train_files, train_folder)

        test_folder = _create_image_class_output_folder(
            output_folders[1],
            image_class_folder)

        _copy_files(test_files, test_folder)

        validation_folder = _create_image_class_output_folder(
            output_folders[2],
            image_class_folder)

        _copy_files(validation_files, validation_folder)

def _get_files_from_folder(folder_path: str) -> list:
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]


def _copy_files(files: list, output_folder: str) -> None:
    for file in files:
        shutil.copy2(file, output_folder)

def _create_output_folder(root_path:Path, folder_name:str) -> Path:
    output_path = os.path.join(root_path, folder_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def _create_image_class_output_folder(root_path:Path, image_class:str) -> Path:
    image_class_path = os.path.join(root_path, image_class)

    if not os.path.exists(image_class_path):
        os.makedirs(image_class_path)

    return image_class_path
