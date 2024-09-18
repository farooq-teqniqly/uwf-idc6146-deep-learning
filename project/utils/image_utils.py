import os.path
from pathlib import Path

from tensorboard.summary.v1 import image


def create_train_test_validation_sets(
        input_dir:Path,
        output_dir:Path,
        filter,
        train_pct:float) -> None:

    output_folders = [_create_output_folder(output_dir, f) for f
                      in ["train", "test", "val"]]

    image_class_folders = [_create_image_class_folder(of, image_class)
                           for of in output_folders
                           for image_class in os.listdir(input_dir)]



def _create_output_folder(root_path:Path, folder_name:str) -> Path:
    output_path = os.path.join(root_path, folder_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def _create_image_class_folder(root_path:Path, image_class:str) -> Path:
    image_class_path = os.path.join(root_path, image_class)

    if not os.path.exists(image_class_path):
        os.makedirs(image_class_path)

    return image_class_path