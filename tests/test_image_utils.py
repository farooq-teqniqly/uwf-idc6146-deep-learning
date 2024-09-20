import os
import shutil
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized

from project.utils.image_utils import split_and_organize_images

JPEG_FILTER = "*.jpeg"


class TestImageUtils(unittest.TestCase):
    def setUp(self):
        self._input_dir = Path("./images224")
        self._output_dir = Path("test_create_train_test_validation_sets")

    def tearDown(self):
        if not os.path.exists(self._output_dir):
            return

        shutil.rmtree(self._output_dir)

    def create_output_dir_if_not_exists(self):
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    @parameterized.expand([
        ("bird", 14, 4, 2),
        ("cat", 14, 4, 2),
    ])
    def test_create_train_test_validation_sets(
            self,
            image_class,
            expected_train_file_count,
            expected_test_file_count,
            expected_validation_file_count):
        self.create_output_dir_if_not_exists()

        split_and_organize_images(self._input_dir, self._output_dir)

        train_folder = Path(os.path.join(self._output_dir, "train", image_class))
        test_folder = Path(os.path.join(self._output_dir, "test", image_class))
        val_folder = Path(os.path.join(self._output_dir, "val", image_class))

        assert self._get_file_count(train_folder) == expected_train_file_count
        assert self._get_file_count(test_folder) == expected_test_file_count
        assert self._get_file_count(val_folder) == expected_validation_file_count

    @parameterized.expand([1, 1.1, 0, -0.01, -1])
    def test_invalid_train_pct_raises_error(
            self,
            train_pct):

        with pytest.raises(ValueError):
            split_and_organize_images(
                self._input_dir,
                self._output_dir,
                train_percentage=train_pct)

    @parameterized.expand([1, 1.1, 0, -0.01, -1])
    def test_invalid_test_pct_raises_error(
            self,
            test_pct):

        with pytest.raises(ValueError):
            split_and_organize_images(
                self._input_dir,
                self._output_dir,
                test_percentage=test_pct)

    @parameterized.expand([0, -1])
    def test_invalid_worker_raises_error(self, max_workers):
        with pytest.raises(ValueError):
            split_and_organize_images(
                self._input_dir,
                self._output_dir,
                max_workers=max_workers)

    @parameterized.expand([
        (0.7, 0.3),
        (0.6, 0.5),
    ])
    def test_train_test_pct_should_be_less_than_1(self, train_pct, test_pct):
        with pytest.raises(ValueError):
            split_and_organize_images(
                self._input_dir,
                self._output_dir,
                train_percentage=train_pct,
                test_percentage=test_pct)

    def _get_file_count(self, folder:Path) -> int:
        return len(list(folder.rglob(JPEG_FILTER)))

if __name__ == "__main__":
    unittest.main()
