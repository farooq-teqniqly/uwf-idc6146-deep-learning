import os
import pickle
import tempfile
import unittest

import pytest
import tensorflow as tf

from project.utils.create_dataset import save_dataset


class SaveDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.images_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255)
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_save_dataset(self):
        output_file = os.path.join(self.tempdir.name, "output.pkl")
        save_dataset(self.tempdir.name, output_file)

    def test_load_dataset(self):
        with open("imagenet_224.pkl", "rb") as f:
            data_to_save = pickle.load(f)

        required_keys = ["train_images", "train_labels", "batch_image_files"]

        for key in required_keys:
            assert key in data_to_save
            assert len(data_to_save[key]) > 0

    def test_dir_not_existing(self):
        output_file = os.path.join(self.tempdir.name, "output.pkl")

        with pytest.raises(Exception):
            save_dataset("nonexistent", output_file)

    def test_wrong_output_file_path(self):
        output_file = os.path.join(self.tempdir.name, "nonexistent", "output.pkl")

        with pytest.raises(Exception):
            save_dataset(self.tempdir.name, output_file)


if __name__ == "__main__":
    unittest.main()
