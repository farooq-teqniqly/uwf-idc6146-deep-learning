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
from sklearn.model_selection import train_test_split

import tensorflow as tf


def save_dataset(
        input_dir: Path,
        output_file: Path,
        train_size=0.7,
        val_size=0.2) -> None:
    """
    Save a dataset of images and their corresponding labels to a file.
    Args:
        input_dir (Path): The directory containing the image dataset.
        output_file (Path): The file path where the dataset will be saved.
        train_size (float): Proportion of the dataset to include in the training set.
        val_size (float): Proportion of the dataset to include in the validation set.
    Raises:
        ValueError: If the input directory does not exist or if sizes are incorrect.
        IOError: If there is an error generating image data from the input directory
                 or saving data to the output file.
    The function performs the following steps:
    1. Verifies that the input directory exists.
    2. Initializes an image data generator with rescaling.
    3. Generates image data from the input directory.
    4. Iterates over the dataset and collects images, labels, and filenames.
    5. Splits the data into training, validation, and test sets.
    6. Stores the collected data in a dictionary.
    7. Saves the dictionary to the specified output file using pickle serialization.
    """
    if not input_dir.exists():
        msg = f"The input directory {input_dir} does not exist."
        raise ValueError(msg)
    if not 0 < train_size < 1:
        raise ValueError("Training set percentage must be a value between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("Validation set percentage must be a value between 0 and 1.")
    if train_size + val_size >= 1:
        raise ValueError(
            "The sum of training and validation set percentages cannot exceed or be equal to 1.")

    test_size = 1 - train_size - val_size
    images_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    try:
        generator = images_generator.flow_from_directory(input_dir)
    except Exception as e:
        msg = f"Error generating image data from {input_dir}: {e}"
        raise IOError(msg) from e

    all_images, all_labels, all_filenames = [], [], []
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

    X_train, X_temp, y_train, y_temp, filenames_train, filenames_temp = train_test_split(
        all_images, all_labels, all_filenames, train_size=train_size, random_state=42)

    X_val, X_test, y_val, y_test, filenames_test, filenames_val = train_test_split(
        X_temp, y_temp, filenames_temp, test_size=(test_size / (test_size + val_size)),
        random_state=42)

    data_to_save = dict(
        train_images=X_train,
        train_labels=y_train,
        val_images=X_val,
        val_labels=y_val,
        test_images=X_test,
        test_labels=y_test,
        filenames=dict(
            train=filenames_train,
            val=filenames_val,
            test=filenames_test
        )
    )

    try:
        with open(output_file, 'wb') as file:
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
    required_keys = [
        "train_images",
        "train_labels",
        "test_images",
        "test_labels",
        "val_images",
        "val_labels",
        "filenames",
    ]

    try:
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        if not all(key in data for key in required_keys):
            msg = f"Missing one or more required keys in the loaded data: {file_name}"
            raise KeyError(msg)
        return (data[required_keys[0]],
                data[required_keys[1]],
                data[required_keys[2]],
                data[required_keys[3]],
                data[required_keys[4]],
                data[required_keys[5]],
                data[required_keys[6]],)
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

