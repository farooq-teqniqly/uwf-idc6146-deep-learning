# Deep Learning Project - Running the Code

There are two notebooks in code.zip for the two models mentioned in the final project report.

simple_model.ipynb: This is the simple CNN used as a baseline.
googlenet_model.ipynb: This is the reproduction of the GoogLeNet Inception CNN.
There are also three zip files containing the images used to train and evaluate the models:

imagenet_224.zip: Downscaled subset of the ImageNet dataset used for training.
imagenet_224_eval.zip: Downscaled subset of the ImageNet dataset used for evaluation.
cifar-10.zip: CIFAR-10 dataset used to evaluate the models against a different dataset.
Download the ZIP files from my Google Drive:

https://drive.google.com/drive/folders/1842AsUd3_KbPI7aeR5lpWxIC14mvzM7W?usp=sharing

Ensure the notebooks and zip files are in the same directory.
In order to run the notebooks, do the following:

Create a Python virtual environment using venv (optional, but a good practice):
python -m venv .
Activate the virtual environment:
.\Scripts\activate  
Run pip install -r requirements.txt. This will install Jupyter, Tensorflow, and Matplotlib.
Run jupyter lab which should open a browser and allow you to select the notebooks to run.
Each notebook contains instructions on how to properly unzip the archives containing the images used to train and evaluate the models.