# PyTorch Scholarship Challenge NanoDegree

# Flower classification

A scholarship provided by Facebook and Udacity covering concepts behind deep learning and how to build deep learning models using PyTorch.

## 1. Problem to solve

Build an application to tell the name of flower from an image.

Using convolutional neural network with Transfer Learning I trained an image classifier that is able to identify 102 different flower species with 95% testing accuracy.
This image classifier can be used to identify flower species from new images, e.g., in a phone app that tells you the name of the flower your camera is looking at.

## 2. Available data

[102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) was given by the Nanodegree program. This dataset contains images of 102 different flower species with lables. These images have different sizes.

## 3. What approach I took to implement

[Link to notebook](Image_Classifier_Udacity_Hitesh.ipynb)

1. Data loading and data preprocessing

    - Load image data
    - Training set: apply transformations such as rotation, scaling, and horizontal flipping (model generalizes / performs better)
    - All datasets: Resize and crop to the appropriate image size (required by pre-trained model)
    - All datasets: Normalize image colors (RGB) using mean and standard deviation of pre-trained model
    - Training set: data shuffled at each epoch

2. Build and train the model

    - Load a pre-trained network `densenet121` ([reference](https://arxiv.org/pdf/1608.06993.pdf)) and freeze parameters (Use of transfer Learning)
    - Define a new, untrained neural network as a classifier. The classifier has a hidden layer (ReLU activation) and an output layer (LogSoftmax activation). Assign dropout to reduce overfitting.
    - Assign criterion (NLLLoss, negative log loss) and optimizer (Adam, adaptive moment estimation, [reference](https://arxiv.org/abs/1412.6980))
    - Train the classifier layers using forward and backpropagation on GPU
    - Track the loss and accuracy on the validation set to determine the best hyperparameters

3. Use the trained classifier to predict image content

    - Test trained model on testing set (95% accuracy)
    - Save trained model as checkpoint
    - Write a function that gives top-5 most probable flower names based on image path


<img src="assets/inference_example2.png" width=300>



