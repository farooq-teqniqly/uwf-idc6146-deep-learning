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
        filter = "*.jpeg"

        create_train_test_validation_sets(
            input_dir,
            output_dir,
            filter,
            0.7)

        train_folder = Path(os.path.join(output_dir, "train"))
        files = list(train_folder.rglob(filter))
        # assert len(files) == 14



if __name__ == "__main__":
    unittest.main()
