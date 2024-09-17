import os
import tempfile
import unittest

import pytest
import tensorflow as tf

from project.utils.dataset_utils import load_dataset, save_dataset


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
        (train_images, train_labels, batch_image_files) = load_dataset(
            "imagenet_224.pkl")

        assert len(train_images) > 0
        assert train_images.shape == (32, 256, 256, 3)

        assert len(train_labels) > 0
        assert train_labels.shape == (32, 10)

        assert len(batch_image_files) == 32

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
