import os
import pickle
import random
import string
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

        with open(output_file, "rb") as f:
            data_to_save = pickle.load(f)

        assert "train_images" in data_to_save
        assert "train_labels" in data_to_save
        assert "batch_image_files" in data_to_save

    def test_dir_not_existing(self):
        random_name = "".join(random.choices(
            string.ascii_uppercase + string.digits, k=10))

        output_file = os.path.join(self.tempdir.name, "output.pkl")

        with pytest.raises(Exception):
            save_dataset(random_name, output_file)

    def test_wrong_output_file_path(self):
        output_file = os.path.join(self.tempdir.name, "nonexistent", "output.pkl")

        with pytest.raises(Exception):
            save_dataset(self.tempdir.name, output_file)


if __name__ == "__main__":
    unittest.main()
