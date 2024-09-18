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

        create_train_test_validation_sets(input_dir, output_dir, filter)

        assert os.path.exists(os.path.join(output_dir, "train"))
        assert os.path.exists(os.path.join(output_dir, "val"))
        assert os.path.exists(os.path.join(output_dir, "test"))

if __name__ == "__main__":
    unittest.main()
