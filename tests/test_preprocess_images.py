import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from parameterized import parameterized
from PIL import Image

from project.utils import preprocess_images


class TestPreprocessImages(unittest.TestCase):
    def setUp(self):
        self.log_mock = MagicMock()

    def tearDown(self):
        self.log_mock.reset_mock()

    def create_dummy_image(self, path, size):
        with Image.new("RGB", size) as img:
            img.save(path)

    @parameterized.expand([
        ((32, 32),),
        ((224, 224),),
        ((512, 32),),
        ((32, 512),),
    ])
    def test_resize_image(self, original_size):
        image_path = Path("test_original.jpg")
        self.create_dummy_image(image_path, original_size)

        output_path = Path("test_resized.jpg")
        output_size = (224, 224)
        preprocess_images.resize_image(image_path, output_path, output_size)

        with Image.open(output_path) as img:
            assert img.size == output_size

        os.remove(image_path)
        os.remove(output_path)

    @parameterized.expand([
        (0,),
        (-1,),
    ])
    def test_invalid_workers(self, workers):
        with pytest.raises(ValueError):
            preprocess_images.process_images(
                Path("inputdir"),
                Path("outputdir"),
                (10, 10),
                self.log_mock,
                workers)

    @parameterized.expand([
        ((0, 0),),
        ((-1, 1),),
        ((1, -1),),
    ])
    def test_invalid_size(self, size):
        with pytest.raises(ValueError):
            preprocess_images.process_images(
                Path("inputdir"),
                Path("outputdir"),
                size,
                self.log_mock)



if __name__ == "__main__":
    unittest.main()