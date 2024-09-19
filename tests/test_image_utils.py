import unittest
import os

from project.utils.image_utils import create_train_test_validation_sets
from pathlib import Path

class TestImageUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_train_test_validation_sets(self):
        input_dir = Path("./images224")
        output_dir = Path("test_create_train_test_validation_sets")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filter = "*.jpeg"

        create_train_test_validation_sets(
            input_dir,
            output_dir,
            filter,
            0.7,
        0.2)

        train_folder = Path(os.path.join(output_dir, "train", "bird"))
        test_folder = Path(os.path.join(output_dir, "test", "bird"))
        val_folder = Path(os.path.join(output_dir, "val", "bird"))

        assert os.path.exists(train_folder)
        assert os.path.exists(val_folder)
        assert os.path.exists(test_folder)

        train_file_count = list(train_folder.rglob(filter))
        assert len(train_file_count) == 14

        test_file_count = list(test_folder.rglob(filter))
        assert len(test_file_count) == 4



if __name__ == "__main__":
    unittest.main()
