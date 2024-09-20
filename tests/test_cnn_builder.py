import sys
import unittest
from sys import modules
from unittest.mock import MagicMock, patch

from project.utils.cnn_builder import CNNBuilder


class TestCNNBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = CNNBuilder()

    def test_build(self):
        self.builder.add_input(kernel_size=7)
        self.builder.add_max_pool()
        self.builder.add_convolution(filters=192, kernel_size=3, strides=(1, 1))
        self.builder.add_max_pool()
        self.builder.add_dropout(0.4)
        self.builder.add_flatten()
        self.builder.add_output_layer(10, activation="softmax")

        model = self.builder.build()
        assert len(model.layers) == 7

if __name__ == "__main__":
    unittest.main()
