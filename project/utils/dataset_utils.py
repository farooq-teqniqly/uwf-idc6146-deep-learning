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

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        msg = "Training set percentage must be a value between 0 and 1."
        raise ValueError(msg)
    if not 0 < val_size < 1:
        msg = "Validation set percentage must be a value between 0 and 1."
        raise ValueError(msg)
    if train_size + val_size >= 1:
        msg = ("The sum of training and validation set percentages cannot exceed or "
               "be equal to 1.")
        raise ValueError(msg)

    images_generator = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    try:
        generator = images_generator.flow_from_directory(
            input_dir,
            batch_size=32,
            class_mode="binary")

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

    test_size = 1 - train_size - val_size

    (x_train,
     x_temp,
     y_train,
     y_temp,
     filenames_train,
     filenames_temp) = train_test_split(
        all_images,
        all_labels,
        all_filenames,
        train_size=train_size,
        random_state=42)

    (x_val,
     x_test,
     y_val,
     y_test,
     filenames_test,
     filenames_val) = train_test_split(
        x_temp,
        y_temp,
        filenames_temp,
        test_size=(test_size / (test_size + val_size)),
        random_state=42)

    data_to_save = dict(
        train_images=x_train,
        train_labels=y_train,
        val_images=x_val,
        val_labels=y_val,
        test_images=x_test,
        test_labels=y_test,
        filenames=dict(
            train=filenames_train,
            val=filenames_val,
            test=filenames_test,
        ),
    )

    try:
        with open(output_file, "wb") as file:
            pickle.dump(data_to_save, file)
    except (FileNotFoundError, IOError, pickle.PicklingError) as e:
        msg = f"Error saving data to {output_file}: {e}"
        raise IOError(msg) from e

def load_dataset(file_name: Path) -> tuple:
    """
    Loads a dataset from a file and verifies the integrity of the data.

    Parameters:
    file_name (Path): The path to the file containing the dataset.

    Returns:
    tuple: A tuple containing the train images, train labels, test images, test labels,
    validation images, validation labels, and filenames.

    Raises:
    RuntimeError: If the file cannot be opened, unpickled, or is missing
    any required key.
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

        return tuple(data[key] for key in required_keys)

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

