import numpy as np
import pandas as pd
class Perceptron:
    def __init__(self, input_size, learning_rate, epochs):
    # Initialize weights and bias with random values
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        # Set learning rate and number of epochs
        self.learning_rate = learning_rate
        self.epochs = epochs
    def step_activation(self, x):
        # Step activation function
        return 1 if x >= 0 else 0
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return 1
        
    def predict(self, x):
        # Compute the weighted sum of inputs and bias
        net_input = np.dot(x, self.weights) + self.bias
        # Apply step activation function to the net input
        return self.step_activation(net_input)
    def predict_linear(self, x):
        net_input = np.dot(x, self.weights) + self.bias
        # Apply step activation function to the net input
        return self.linear(net_input)
    def trainSGD(self, X, y):
    # Training loop
        for epoch in range(self.epochs):
        # Initialize total error for the epoch
            total_error = 0
            # Shuffle the training examples for this epoch
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            # Iterate over each shuffled training example
            for i in range(len(X_shuffled)):
                # Make a prediction for the current input
                prediction = self.predict(X_shuffled[i])
                # Compute the error (desired - predicted)
                error = y_shuffled[i] - prediction
                # Update weights and bias using stochastic gradient descent
                self.weights += self.learning_rate * error * X_shuffled[i]
                self.bias += self.learning_rate * error
                # Accumulate the absolute error for the epoch
                total_error += abs(error)
                # Print total errors for each epoch
            print(f"Epoch {epoch + 1}, Total Absolute Error: {total_error}")
    def trainSGD_linear(self, X, y):
        for epoch in range(self.epochs):
        # Initialize total error for the epoch
            total_mse = 0.0
            # Shuffle the training examples for this epoch
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = y[indices]
            # Iterate over each shuffled training example
            for i in range(len(X_shuffled)):
                # Make a prediction for the current input
                prediction = self.predict_linear(X_shuffled[i])
                # Compute the error (desired - predicted)
                error = Y_shuffled[i] - prediction
                # Compute gradients using linear derivative
                delta = error * self.linear_derivative(prediction)
                # Update weights and bias using stochastic gradient descent
                self.weights += self.learning_rate * delta * X_shuffled[i]
                self.bias += self.learning_rate * delta
                # Accumulate the squared error for this example
                total_mse += error ** 2
            # Calculate the mean squared error for this epoch
            mean_mse = total_mse / len(X)
            print(f"Epoch {epoch + 1}/{self.epochs}, Mean Squared Error:{mean_mse:.4f}")
        


