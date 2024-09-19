import os
import shutil
import unittest
from pathlib import Path

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

    def test_create_train_test_validation_sets(self):
        self.create_output_dir_if_not_exists()

        create_train_test_validation_sets(
            input_dir=self._input_dir,
            output_dir=self._output_dir,
            file_filter=JPEG_FILTER,
        )

        train_folder = Path(os.path.join(self._output_dir, "train", "bird"))
        test_folder = Path(os.path.join(self._output_dir, "test", "bird"))
        val_folder = Path(os.path.join(self._output_dir, "val", "bird"))

        assert os.path.exists(train_folder)
        assert os.path.exists(val_folder)
        assert os.path.exists(test_folder)

        train_file_count = list(train_folder.rglob(JPEG_FILTER))
        assert len(train_file_count) == TRAIN_FILE_COUNT

        test_file_count = list(test_folder.rglob(JPEG_FILTER))
        assert len(test_file_count) == TEST_FILE_COUNT

        val_file_count = list(val_folder.rglob(JPEG_FILTER))
        assert len(val_file_count) == VAL_FILE_COUNT

if __name__ == "__main__":
    unittest.main()
