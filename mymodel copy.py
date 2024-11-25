{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def sigmoid(z):\n",
    "    return 1 / (1 + np.exp(-z))\n",
    "\n",
    "# Logistic Regression class\n",
    "class LogisticRegressionScratch:\n",
    "    def __init__(self, learning_rate=0.01, num_iterations=1000):\n",
    "        self.learning_rate = learning_rate\n",
    "        self.num_iterations = num_iterations\n",
    "        self.weights = None\n",
    "        self.bias = None\n",
    "\n",
    "    # Fit the model to the training data\n",
    "    def fit(self, X, y):\n",
    "        # Initialize parameters\n",
    "        num_samples, num_features = X.shape\n",
    "        self.weights = np.zeros(num_features)\n",
    "        self.bias = 0\n",
    "\n",
    "        # Gradient Descent\n",
    "        for _ in range(self.num_iterations):\n",
    "            # Linear model\n",
    "            linear_model = np.dot(X, self.weights) + self.bias\n",
    "            # Apply sigmoid function\n",
    "            y_predicted = sigmoid(linear_model)\n",
    "\n",
    "            # Compute gradients\n",
    "            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))\n",
    "            db = (1 / num_samples) * np.sum(y_predicted - y)\n",
    "\n",
    "            # Update parameters\n",
    "            self.weights -= self.learning_rate * dw\n",
    "            self.bias -= self.learning_rate * db\n",
    "\n",
    "    # Predict probabilities for X\n",
    "    def predict_proba(self, X):\n",
    "        linear_model = np.dot(X, self.weights) + self.bias\n",
    "        return sigmoid(linear_model)\n",
    "\n",
    "    # Predict binary labels for X\n",
    "    def predict(self, X, threshold=0.5):\n",
    "        y_predicted_probs = self.predict_proba(X)\n",
    "        return [1 if i > threshold else 0 for i in y_predicted_probs]\n",
    "\n",
    "# Example usage with dummy data\n",
    "# Let's create some example data to try this out\n",
    "X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Features\n",
    "y = np.array([0, 0, 1, 1])  # Target labels"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ipykernel_py3.8",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.20"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
