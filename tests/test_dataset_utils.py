import os
import tempfile
import unittest
from pathlib import Path

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
        input_dir = Path("./images")
        output_file = Path(os.path.join(self.tempdir.name, "test_save_dataset.pkl"))
        save_dataset(input_dir, output_file)

        assert os.path.exists(output_file)

    def test_load_dataset(self):
        input_file = Path("test_load_dataset.pkl")
        (
            train_images,
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
            filenames) = load_dataset(input_file)

        assert len(train_images) == 28
        assert len(test_images) == 8
        assert len(val_images) == 4

        assert len(train_labels) == 28
        assert len(test_labels) == 8
        assert len(val_labels) == 4

        assert len(filenames["train"]) == 28
        assert len(filenames["test"]) == 8
        assert len(filenames["val"]) == 4

    def test_dir_not_existing(self):
        input_dir = Path("nonexistent")
        output_file = Path(os.path.join(self.tempdir.name, "output.pkl"))

        with pytest.raises(ValueError):
            save_dataset(input_dir, output_file)

    def test_wrong_output_file_path(self):
        input_dir = Path(self.tempdir.name)
        output_file = Path(os.path.join(self.tempdir.name, "nonexistent", "output.pkl"))

        with pytest.raises(IOError):
            save_dataset(input_dir, output_file)


if __name__ == "__main__":
    unittest.main()
