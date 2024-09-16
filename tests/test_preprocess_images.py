import os
import unittest
from pathlib import Path

from parameterized import parameterized
from PIL import Image

from project.utils import preprocess_images


class TestPreprocessImages(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
