"""
Module: image_utils

This module contains utilities for creating train, test, and validation sets
from a directory of images. The sets are created based on given percentages
for training and testing, with the remainder being used for validation.

Functions:
    create_train_test_validation_sets: Main function to create the dataset splits.
    _copy_files_to_output: Helper function to copy files to the output directory.
    _split_files: Helper function to split the files into train, test, and
    validation sets.
    _get_source_files: Helper function to retrieve source files from a directory.
    _get_files_from_folder: Helper function to get files from a specific folder.
    _copy_files: Helper function to copy files to the specified output folder.
    _create_output_folder: Helper function to create an output directory if it
    doesn't exist.
    _create_image_class_output_folder: Helper function to create sub-directories within
    the output directory.
"""

import os.path
import random
import shutil
from pathlib import Path
from typing import List, Tuple

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
VAL_FOLDER = "val"
SHUFFLE_ROUNDS = 3


def create_train_test_validation_sets(
        input_dir: Path,
        output_dir: Path,
        file_filter="*.jpeg",
        train_percentage: float = 0.7,
        test_percentage: float = 0.2) -> None:
    """
    Creates train, test, and validation sets from the given input directory and outputs
    them to the specified output directory.

    Parameters:
        input_dir (Path): The directory containing the input images organized in
        subdirectories per class.
        output_dir (Path): The directory where the train, test, and validation sets
        will be saved.
        file_filter (str): The glob pattern to filter files in the input directory.
        train_percentage (float): The percentage of images to be used for the
        training set.
        test_percentage (float): The percentage of images to be used for the
        testing set.

    Returns:
        None
    """
    output_folders = [_create_output_folder(output_dir, folder_name)
                      for folder_name in [TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER]]

    image_class_folders = os.listdir(input_dir)

    for image_class_folder in image_class_folders:
        files = _get_source_files(image_class_folder, input_dir)
        random.shuffle(files)

        train_files, test_files, validation_files = _split_files(
            files,
            test_percentage,
            train_percentage)

        _copy_files_to_output(output_folders[0], image_class_folder, train_files)
        _copy_files_to_output(output_folders[1], image_class_folder, test_files)
        _copy_files_to_output(output_folders[2], image_class_folder, validation_files)


def _copy_files_to_output(
        output_folder: Path,
        image_class_folder: str,
        files: List[str]) -> None:
    folder = _create_image_class_output_folder(
        output_folder,
        image_class_folder)

    _copy_files(files, folder)


def _split_files(
        files: List[str],
        test_percentage: float,
        train_percentage: float) -> Tuple[List[str], List[str], List[str]]:
    train_count = int(len(files) * train_percentage)
    test_count = int(len(files) * test_percentage)

    train_files = files[:train_count]
    test_files = files[train_count:train_count + test_count]
    validation_files = files[train_count + test_count:]

    return train_files, test_files, validation_files


def _get_source_files(image_class_folder: str, input_dir: Path) -> List[str]:
    source_folder_path = os.path.join(input_dir, image_class_folder)
    return _get_files_from_folder(source_folder_path)


def _get_files_from_folder(folder_path: str) -> List[str]:
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]


def _copy_files(files: list, output_folder: str) -> None:
    for file in files:
        shutil.copy2(file, output_folder)


def _create_output_folder(root_path: Path, folder_name: str) -> Path:
    output_path = os.path.join(root_path, folder_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path


def _create_image_class_output_folder(root_path: Path, image_class: str) -> Path:
    image_class_path = os.path.join(root_path, image_class)

    if not os.path.exists(image_class_path):
        os.makedirs(image_class_path)

    return image_class_path
