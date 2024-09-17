"""
dataset_utils.py

This module contains utility functions for saving and loading datasets, specifically
tailored for handling large datasets used in machine learning applications.
The primary functions provided in this module are:

- `save_dataset`: Saves the dataset to a specified file in a compressed format
for efficient storage.
- `load_dataset`: Loads the dataset from a specified file, reconstructing
the structure necessary for training models.

These functions facilitate the seamless handling of datasets, ensuring that
they can be easily saved and retrieved without loss or corruption.

Functions:
    save_dataset: Saves the given dataset to a file.
    load_dataset: Loads the dataset from a file.
    main: A main function to demonstrate the usage of the utility
    functions or for standalone script execution.

Attributes:
    tf: TensorFlow module import, if applicable.
"""
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf


def save_dataset(input_dir:Path, output_file:Path) -> None:
    """
    Save a dataset of images and their corresponding labels to a file.

    Args:
        input_dir (Path): The directory containing the image dataset.
        output_file (Path): The file path where the dataset will be saved.

    Raises:
        ValueError: If the input directory does not exist.
        IOError: If there is an error generating image data from the input directory
        or saving data to the output file.

    The function performs the following steps:
    1. Verifies that the input directory exists.
    2. Initializes an image data generator with rescaling.
    3. Generates image data from the input directory.
    4. Iterates over the dataset and collects images, labels, and filenames.
    5. Stores the collected data in a dictionary.
    6. Saves the dictionary to the specified output file using pickle serialization.
    """

    if not input_dir.exists():
        msg = f"The input directory {input_dir} does not exist."
        raise ValueError(msg)

    images_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    try:
        generator = images_generator.flow_from_directory(input_dir)
    except Exception as e:
        msg = f"Error generating image data from {input_dir}: {e}"
        raise IOError(msg) from e

    all_images = []
    all_labels = []
    all_filenames = []

    for _ in range(generator.samples // generator.batch_size + 1):
        try:
            images, labels = next(generator)
            filenames = generator.filenames
            batch_indices = generator.index_array[:len(images)]
            batch_filenames = [filenames[idx] for idx in batch_indices]

            all_images.extend(images)
            all_labels.extend(labels)
            all_filenames.extend(batch_filenames)
        except StopIteration:
            break

    batch_image_files = {all_filenames[i]: all_images[i]
                         for i in range(len(all_images))}
    data_to_save = dict(
        train_images=all_images,
        train_labels=all_labels,
        batch_image_files=batch_image_files,
    )

    try:
        with open(output_file, "wb") as file:
            pickle.dump(data_to_save, file)
    except (FileNotFoundError, IOError, pickle.PicklingError) as e:
        msg = f"Error saving data to {output_file}: {e}"
        raise IOError(msg) from e


def load_dataset(file_name: Path) -> Tuple[list, list, List[str]]:
    """
    Loads a dataset from a pickle file.

    Arguments:
    file_name: Path to the file containing the dataset.

    Returns:
    A tuple containing three elements:
    - List of training images.
    - List of training labels.
    - List of batch image file names.

    Raises:
    RuntimeError: If the file cannot be opened, read, or unpickled;
    or if required keys are missing from the dataset.
    """
    required_keys = ["train_images", "train_labels", "batch_image_files"]

    try:
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        if not all(key in data for key in required_keys):
            msg = f"Missing one or more required keys in the loaded data: {file_name}"
            raise KeyError(msg)
        return data[required_keys[0]], data[required_keys[1]], data[required_keys[2]]
    except (FileNotFoundError, IOError) as e:
        msg = f"Error opening the file {file_name}: {e}"
        raise RuntimeError(msg) from e
    except pickle.UnpicklingError as e:
        msg = f"Error unpickling the file {file_name}: {e}"
        raise RuntimeError(msg) from e
    except KeyError as e:
        msg = f"Missing one or more required keys in the loaded data: {file_name}"
        raise RuntimeError(msg) from e

def main():
    parser = argparse.ArgumentParser(
        description="Pickle a dataset representing the ImageNet images.")

    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the input folder containing images")

    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to the output pickle file")

    args = parser.parse_args()
    save_dataset(args.input_dir, args.output_file)

    # (train_images, train_labels, batch_image_files) = load_dataset(pickle_file)


if __name__ == "__main__":
    main()

