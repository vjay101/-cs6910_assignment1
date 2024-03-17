# -cs6910_assignment1
This repository contains the implementation of Feed Forward Neural Network from scratch using numpy and pandas.

Description: Inside the repository, you'll find the implementation of a feedforward neural network designed to tackle a multiclass classification problem using the Fashion-MNIST dataset. The repository includes the following files:

- `train.py`: This file encompasses the entire implementation of the neural network.
- `question1.py`: Here, you'll find the code responsible for displaying all the labels along with their corresponding images.
- `sweep.txt`: Contains the configuration used for hyperparameter tuning.
- `Sample images`: Holds all sample images across all classes.

Best Configuration Achieved:

- Loss Function: Cross Entropy
- Number of Hidden Layers: 2
- Size of Hidden Layers: 128
- Batch Size: 64
- Learning Rate: 0.01
- Activation Function: Sigmoid
- Weight Initialization: Xavier
- Optimizer: Nadam
- Weight Decay: 0.0001
- Epochs: 10

Summarized Results:

- Train Accuracy: 89.3%
- Train Loss: 0.4278
- Validation Accuracy: 88.7%
- Validation Loss: 0.4707
- Test Accuracy: 86.92%
