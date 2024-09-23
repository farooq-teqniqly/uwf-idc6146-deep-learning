from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from typing import Tuple, List

class CNNBuilder:
    def __init__(self):
        self._model = Sequential()

    def add_input(
            self,
            filters:int=64,
            kernel_size:int=3,
            strides:Tuple[int, int]=(2, 2),
            activation:str="relu",
            input_shape:List[int]=[224,224,3]) -> None:

        self._model.add(Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            input_shape=input_shape))

    def add_max_pool(
            self,
            pool_size:Tuple[int,int]=(3, 3),
            strides:int=2,
            padding:str="same") -> None:

        self._model.add(MaxPool2D(
            pool_size=pool_size, strides=strides, padding=padding))

    def add_convolution(
            self,
            filters: int = 64,
            kernel_size: int = 3,
            strides: Tuple[int, int] = (2, 2),
            activation: str = "relu",
            padding:str="same") -> None:

        self._model.add(Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding=padding))

    def add_flatten(self):
        self._model.add(Flatten())

    def add_dropout(self, rate:float):
         self._model.add(Dropout(rate))

    def add_full_connection(self, units:int=128, activation:str="relu"):
        self._model.add(Dense(units=units, activation=activation))

    def add_output_layer(self, num_classes:int, activation:str="relu"):
        self._model.add(Dense(units=num_classes, activation=activation))

    def build(self):
        self._model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")])

        return self._model