"""
This module handles the loading and saving of image data for training purposes.

It uses TensorFlow's Keras preprocessing utilities to generate batches of
images from a directory,
and then saves that data to a pickle file for later use. Functions are also
provided to load this
saved dataset back into memory.

Classes and functions included:
- `load_dataset`: Loads a dataset from a pickle file and returns the images,
labels, and filenames.
"""
import argparse
import os
import pickle

import tensorflow as tf


def save_dataset(input_dir, output_file):
    """
        Saves a dataset into a file.
        This function processes image data from the specified input directory,
        normalizes the images, and saves them along with their labels and
        filenames into a specified output file using pickle.
        Args:
            input_dir (str): The path to the directory containing the images.
            output_file (str): The path where the processed dataset will be saved.
        Raises:
            ValueError: If the input directory does not exist.
            IOError: If there is an error processing images or saving data.
        """
    if not os.path.isdir(input_dir):
        msg = f"The input directory {input_dir} does not exist."
        raise ValueError(msg)

    images_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    generator = images_generator.flow_from_directory(input_dir)

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
    except (IOError, pickle.PicklingError) as e:
        msg = f"Error saving data to {output_file}: {e}"
        raise IOError(msg) from e

def load_dataset(file_name):
    """

    load_dataset(file_name)

    Loads a dataset from a specified file.

    Parameters:
    file_name (str): The path to the dataset file.

    Returns:
    tuple: A tuple containing three elements:
        - train_images: The training images.
        - train_labels: The labels for the training images.
        - batch_image_files: The batch image files.
    """
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data["train_images"], data["train_labels"], data["batch_image_files"]

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

