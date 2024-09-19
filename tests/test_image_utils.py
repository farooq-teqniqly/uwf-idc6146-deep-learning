import unittest
import os
import shutil

from project.utils.image_utils import create_train_test_validation_sets
from pathlib import Path

class TestImageUtils(unittest.TestCase):

    def setUp(self):
        self._input_dir = Path("./images224")
        self._output_dir = Path("test_create_train_test_validation_sets")

    def tearDown(self):
        shutil.rmtree(self._output_dir)

    def test_create_train_test_validation_sets(self):
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        filter = "*.jpeg"

        create_train_test_validation_sets(
            self._input_dir,
            self._output_dir,
            filter,
            0.7,
        0.2)

        train_folder = Path(os.path.join(self._output_dir, "train", "bird"))
        test_folder = Path(os.path.join(self._output_dir, "test", "bird"))
        val_folder = Path(os.path.join(self._output_dir, "val", "bird"))

        assert os.path.exists(train_folder)
        assert os.path.exists(val_folder)
        assert os.path.exists(test_folder)

        train_file_count = list(train_folder.rglob(filter))
        assert len(train_file_count) == 14

        test_file_count = list(test_folder.rglob(filter))
        assert len(test_file_count) == 4



if __name__ == "__main__":
    unittest.main()
