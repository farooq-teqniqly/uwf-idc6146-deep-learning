import os
import shutil
import unittest
from pathlib import Path

from parameterized import parameterized

from project.utils.image_utils import create_train_test_validation_sets

TRAIN_FILE_COUNT = 14
TEST_FILE_COUNT = 4
VAL_FILE_COUNT = 2
JPEG_FILTER = "*.jpeg"


class TestImageUtils(unittest.TestCase):
    def setUp(self):
        self._input_dir = Path("./images224")
        self._output_dir = Path("test_create_train_test_validation_sets")

    def tearDown(self):
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

        create_train_test_validation_sets(self._input_dir, self._output_dir)

        train_folder = Path(os.path.join(self._output_dir, "train", image_class))
        test_folder = Path(os.path.join(self._output_dir, "test", image_class))
        val_folder = Path(os.path.join(self._output_dir, "val", image_class))

        assert self._get_file_count(train_folder) == expected_train_file_count
        assert self._get_file_count(test_folder) == expected_test_file_count
        assert self._get_file_count(val_folder) == expected_validation_file_count

    def _get_file_count(self, folder:Path) -> int:
        return len(list(folder.rglob(JPEG_FILTER)))

if __name__ == "__main__":
    unittest.main()
