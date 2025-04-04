# MNIST Digit Classifier

This project demonstrates how to classify handwritten digits from the MNIST dataset using a neural network built with TensorFlow and Keras. The model is trained on the MNIST training set and evaluated on the validation set. The training and validation loss, as well as accuracy, are plotted during training.

## Project Overview
This deep learning project involves the use of a simple fully connected neural network to classify digits from the MNIST dataset. It uses a `Sequential` model from TensorFlow/Keras and is trained with the Adam optimizer. The model's performance is evaluated with sparse categorical cross-entropy loss and accuracy.

## Dataset
The dataset used is the MNIST dataset, which consists of 60,000 training images and 10,000 validation images of handwritten digits (0-9). Each image is 28x28 pixels in size, and each pixel is grayscale.

## Model Architecture
The model used is a feedforward neural network (fully connected layers):
1. **Flatten**: Converts each 28x28 image into a 1D array of 784 pixels.
2. **Dense Layer**: The first fully connected layer with 32 units and ReLU activation.
3. **Dense Layer**: The second fully connected layer with 32 units and ReLU activation.
4. **Dense Layer**: The output layer with 10 units, corresponding to the 10 possible digit classes (0-9), using Softmax activation.

## Technologies Used
- Python
- TensorFlow / Keras
- Matplotlib

## Installation and Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mnist-digit-classifier.git
    cd mnist-digit-classifier
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the script to train the model and visualize the results:
    ```bash
    python mnist_digit_classifier.py
    ```

## Usage
- The script trains the model on the MNIST training dataset and evaluates it on the validation dataset.
- It plots the training and validation loss, as well as the training and validation accuracy, for each epoch during training.
  
## Example Output
After running the script, you will see two plots:
- The first plot shows the **training loss** and **validation loss** over epochs.
- The second plot shows the **training accuracy** and **validation accuracy** over epochs.

## Contributions
Feel free to contribute! You can fork the repository and submit a pull request with your improvements. Please ensure that your changes follow the code style and pass all tests.
