# Install the Weights & Biases library
!pip install wandb -qU

# Import necessary libraries
import wandb
import os
from keras.datasets import fashion_mnist

# Set the notebook name for W&B logging
os.environ['WAND_NOTEBOOK_NAME'] = 'question1'

# Log in to your W&B account
wandb.login(key="aeb76f579a615642cf7a90ceffa396e4ac213fd8")

# Load Fashion-MNIST dataset
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# Initialize W&B run
wandb.init(project='Deep Learning', entity='cs22m094', name='question1')

# Define categories for labels
categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Collect one sample image for each class
collection = []
label = []
for i in range(len(X_train)):
    if len(label) == 10:
        break
    if categories[Y_train[i]] in label:
        continue
    else:
        collection.append(X_train[i])
        label.append(categories[Y_train[i]])

# Log sample images with corresponding labels to W&B
wandb.log({"Question 1-Sample Images on Fashion Mnist": [wandb.Image(img, caption=lbl) for img, lbl in zip(collection, label)]})

