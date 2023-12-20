from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

def mlp(X, y, hidden_layer_sizes=(10,), max_iter=100, learning_rate=0.01, activation='relu'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create an MLPClassifier with one hidden layer of 10 neurons
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate, activation=activation)
    # Train the MLPClassifier
    mlp.fit(X_train, y_train)
    # make prediction on the testing part
    y_pred = mlp.predict(X_test)
    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    # Plot the loss curve
    plt.plot(mlp.loss_curve_, marker='o', label='Train Loss')
    plt.title('Loss Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Run the MLP function
mlp(X, y)

