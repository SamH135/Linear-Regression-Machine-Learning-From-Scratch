import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, learning_rate=0.0000001, num_iterations=6500, tolerance=1e-3):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.theta = []
        self.cost_history = []



    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: The input data.
            y: The output data.

        Returns:
            None.
        """
        # Initialize parameters (theta) with random values
        np.random.seed(0)  # For reproducibility
        self.theta = np.random.rand(X.shape[1])

        m = len(y)  # Number of training examples

        for i in range(self.num_iterations):
            # gradient = self.gradient(X, y)
            # self.theta -= self.learning_rate * gradient
            # self.cost_history.append(self.cost(X, y))
            # if np.linalg.norm(gradient) < self.tolerance:
            #     break

            # Calculate predictions
            predictions = np.dot(X, self.theta)

            # Calculate the error
            error = predictions - y
            # print("error",error.shape)

            # Calculate gradients
            gradients = (1/m) * np.dot(X.T, error)
            # print("gradients",gradients.shape)

            # Update parameters (theta)
            self.theta -= self.learning_rate * gradients
            if np.all(np.abs(self.theta) <= self.tolerance):
              break



    def predict(self, X):
        """
        Predict the output for the given data.

        Args:
            X: The input data.

        Returns:
            The predicted output.
        """
        return X.dot(self.theta)

    def gradient(self, X, y):
        """
        Calculate the gradient of the cost function.

        Args:
            X: The input data.
            y: The output data.

        Returns:
            The gradient of the cost function.
        """
        return 2 * X.T.dot(self.hypothesis(X) - y)

    def cost(self, X, y):
        """
        Calculate the cost function.

        Args:
            X: The input data.
            y: The output data.

        Returns:
            The cost function.
        """
        return np.mean((self.hypothesis(X) - y)**2)

    def hypothesis(self, X):
        """
        Calculate the hypothesis for the given data.

        Args:
            X: The input data.

        Returns:
            The hypothesis.
        """
        return X.dot(self.theta)

    def train_test_split(self, X, y):
      # convert dataFrame to numpy array
      data = df.to_numpy()

      # Define the split ratio (80/20 in this case)
      split_ratio = 0.8

      # Calculate the number of samples for training and testing
      num_samples = data.shape[0]
      num_train_samples = int(split_ratio * num_samples)
      num_test_samples = num_samples - num_train_samples

      # Shuffle the data randomly
      np.random.shuffle(data)# convert dataFrame to numpy array
      data = df.to_numpy()

      # Define the split ratio (80/20 in this case)
      split_ratio = 0.8

      # Calculate the number of samples for training and testing
      num_samples = data.shape[0]
      num_train_samples = int(split_ratio * num_samples)
      num_test_samples = num_samples - num_train_samples

      # Shuffle the data randomly
      np.random.shuffle(data)

      # Split the features and target labels into training and test sets
      X_train = X[:num_train_samples, :]
      y_train = y[:num_train_samples]
      X_test = X[num_train_samples:, :]
      y_test = y[num_train_samples:]

      return X_train, y_train, X_test, y_test






if __name__ == "__main__":
    # Load the csv file into a DataFrame from github
    df = pd.read_csv("https://github.com/fayez-max/Real-Estate-Data-Set/raw/main/Real%20estate%20valuation%20data%20set.csv")

    # Remove rows with missing values
    df = df.dropna()

    # Check for and remove duplicate rows
    df = df.drop_duplicates()

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))  # Define the size of the figure

    # Customize the color map by specifying cmap
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

    # Add labels and title
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title('Correlation Heatmap')

    plt.show()  # Display the heatmap

    # Select features with the strongest correlation
    selected_features = ['X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

    # Create X with the selected features
    X = df[selected_features].to_numpy()

    # Select the target variable
    y = df['Y house price of unit area'].to_numpy()

    # Create the model
    model = LinearRegression(learning_rate=0.0000001, num_iterations=6500, tolerance=1e-3)

    # split data for training and testing
    X_train, y_train, X_test, y_test = model.train_test_split(X, y)

    # Fit the model to the data to generate theta values
    model.fit(X_train, y_train)

    predicted_price = np.dot(X_test, model.theta)

    # run MSE function
    mse = mean_squared_error(y_test, predicted_price)

    # Print the MSE value
    print(f"MSE: {mse:.2f}")

    # Calculate R^2
    y_predicted = np.dot(X_test, model.theta)

    # Print the R^2 value
    r_2 = 1 - (sum((y_test - y_predicted)**2) / sum((y_test - np.mean(y_test))**2))
    print(f"R^2: {r_2:.2f}")




    # Plot the actual vs. predicted values
    plt.figure(figsize=(8, 6))  # Set the figure size for better visualization
    plt.scatter(y_test, y_predicted, alpha=0.6, color='b', edgecolor='k', label='Actual vs. Predicted')
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title("Actual vs. Predicted Values", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)

    # Add a diagonal line for reference (perfect predictions)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', linewidth=2, label='Perfect Predictions')

    # Add a best-fit line
    m, b = np.polyfit(y_test, y_predicted, 1)  # Fit a linear regression line
    plt.plot(y_test, m * y_test + b, color='g', linewidth=2, label='Best-Fit Line')

    plt.legend(loc='upper left', fontsize=12)

    plt.show()