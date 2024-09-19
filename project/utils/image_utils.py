import os.path
import random
import shutil

from pathlib import Path

from tensorboard.summary.v1 import image


def create_train_test_validation_sets(
        input_dir:Path,
        output_dir:Path,
        filter,
        train_pct:float) -> None:

    output_folders = [_create_output_folder(output_dir, f) for f
                      in ["train", "test", "val"]]

    source_image_class_folders = os.listdir(input_dir)

    image_class_output_folders = [_create_image_class_output_folder(of, image_class)
                                  for of in output_folders
                                  for image_class in source_image_class_folders]

    for source_image_class_folder in source_image_class_folders:
        source_image_class_folder_full_path = os.path.join(input_dir, source_image_class_folder)

        files =[os.path.join(source_image_class_folder_full_path, fn)
                for fn in os.listdir(source_image_class_folder_full_path)]

        [random.shuffle(files) for _ in range(0, 3)]

        file_count = len(files)
        train_file_count = int(file_count * train_pct)
        train_files = files[:train_file_count]

        for file in train_files:
            shutil.copy2(file, image_class_output_folders[0])


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