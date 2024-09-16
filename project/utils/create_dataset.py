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
import pickle

import tensorflow as tf


def save_dataset(input_dir, output_file):
    """
    Saves a dataset into a file.

    This function processes image data from the specified input directory, normalizes
    the images, and saves them along with their labels and filenames into a
    specified output file using pickle.

    Args:
        input_dir (str): The path to the directory containing the images.
        output_file (str): The path where the processed dataset will be saved.

    Raises:
        Exception: An error occurred while processing the images or saving the data.
    """
    images_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    generator = images_generator.flow_from_directory(input_dir)
    train_images, train_labels = next(generator)
    filenames = generator.filenames
    batch_indices = generator.index_array[:len(train_images)]
    batch_filenames = [filenames[idx] for idx in batch_indices]
    batch_image_files = {batch_filenames[i]: train_images[i]
                         for i in range(len(train_images))}

    data_to_save = dict(
        train_images=train_images,
        train_labels=train_labels,
        batch_image_files=batch_image_files,
    )

    with open(output_file, "wb") as file:
        pickle.dump(data_to_save, file)

def load_dataset():
    """
        Loads a dataset from a pickle file.

        Reads the "dataset.pkl" file and loads its contents using the pickle module.

        Returns:
            tuple: Contains `train_images`, `train_labels`, and `batch_image_files`.
    """
    with open("dataset.pkl", "rb") as file:
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

if __name__ == "__main__":
    main()

