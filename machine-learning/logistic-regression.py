# Date: 2026-03-19

# Logistic Regression requires 5 parts
# 1. Sigmoid functiont obtain probability of class
# 2. Cost function to sum up the error of loss function
# 3. Gradient function
# 4. Gradient descent function
# 5. Prediction function (with decision boundary threshold?)


import numpy as np 

# ─────────────────────────────────────────────
# Implementation 1 - Easy to understand 
# ─────────────────────────────────────────────
class LogisticRegressionDirect():
    def __init__(self):
        pass
    
    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))
    
    def cost_function(self, X, y, w, b):
        
        m = len(X)

        # variable to sum the total cost
        cost_sum = 0
        
        # Looping sample by sample
        for i in range(m):
            prediction = np.dot(w, X[i]) + b # simple linear regression
            
            g = self.sigmoid(prediction) #convert to probability
            
            cost = -y[i] * np.log(g) - (1 - y[i]) * np.log(1 - g) #calculate the cost for that specific sample
            
            cost_sum += cost #add that cost to the accumulated cost
        
        return round((1/m) * cost_sum,2)
    
    def calcutate_gradient(self, X, y, w, b):
        n_of_samples, n_of_features = X.shape

        grad_w = np.zeros(n_of_features) # Dependent on count of features
        grad_b = 0 # Since it is just a constant, does not depend on count of features
        
        # For every sample, calculate their w and b
        for i in range(n_of_samples):
            prediction = np.dot(w, X[i]) + b # simple linear regression
            
            g = self.sigmoid(prediction)
            
            # 1 Calculate gradient for bias
            grad_b += g-y[i]
            
            # 2 Calculate gradient for weight
            for j in range(n_of_features):
                grad_w[j] += (g - y[i]) * X[i, j]
        
        grad_w = (1/n_of_samples) * grad_w
        grad_b = (1/n_of_samples) * grad_b
        
        return grad_w, grad_b
    
    def gradient_descent(self, X, y, lr, n_iterations):
        n_of_samples, n_of_features = X.shape
        
        # initialise w and b
        w = np.zeros(n_of_features)
        b = 0
        
        for i in range(n_iterations):
            grad_w, grad_b = self.calcutate_gradient(X, y, w, b)
            
            w -= lr * grad_w
            b -= lr * grad_b
            
            if i % 1000 == 0:
                print(f"Iteration {i} | Cost Function {self.cost_function(X, y, w, b)}")
            
        return w, b
    
    def predict(self, X, w, b, threshold):
        # Initialise prediction as all 0
        n_of_samples, n_of_features = X.shape
        
        preds = np.zeros(n_of_samples)
        
        for i in range(n_of_samples):
            prediction = np.dot(w, X[i]) + b
            g = self.sigmoid(prediction)
            
            # Re-assign the prediction into 1 if it exceeds the threshold
            preds[i] = 1 if g > threshold else 0
        
        return preds # return prediction as array


# ─────────────────────────────────────────────
# Implementation 2 - Optimised 
# ─────────────────────────────────────────────

class LogisticRegressionOptimised():
    def __init__(self, lr=0.01, n_iterations=10000, threshold=0.5):
        # Hyperparameters set at construction
        self.lr = lr
        self.n_iterations = n_iterations
        self.threshold = threshold
        
        # Model parameters, don't exist until fit() is called
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _linear(self, X):
        # Centralised: no more repeating np.dot(w, X[i]) + b everywhere
        return X @ self.w + self.b

    def _cost(self, X, y):
        g = self._sigmoid(self._linear(X))
        g = np.clip(g, 1e-15, 1 - 1e-15)  # prevent log(0)
        return -np.mean(y * np.log(g) + (1 - y) * np.log(1 - g))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iterations):
            g = self._sigmoid(self._linear(X))      # vectorised, no for loop needed
            error = g - y

            grad_w = (1 / n_samples) * X.T @ error  # vectorised gradient
            grad_b = (1 / n_samples) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if i % 1000 == 0:
                print(f"Iteration {i} | Cost: {self._cost(X, y):.4f}")

    def predict_proba(self, X):
        return self._sigmoid(self._linear(X))

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)
    

# ─────────────────────────────────────────────
# Test 
# ─────────────────────────────────────────────

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_raw, y = make_classification(
        n_samples=400,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )

n_sample = X_raw.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

LR_Direct = LogisticRegressionDirect()
LR_Optimised = LogisticRegressionOptimised(threshold=0.3)

########################################

lr = 0.01
n_iterations = 10000

w, b = LR_Direct.gradient_descent(X_train, y_train, lr, n_iterations)

preds = LR_Direct.predict(X_test, w, b, 0.3)

accuracy = np.mean(preds == y_test)

print(f"{LR_Direct.__class__.__name__} | Sample {n_sample} | Accuracy {accuracy:.5f}")

########################################
lr = 0.01
n_iterations = 10000

LR_Optimised.fit(X_train, y_train)

preds = LR_Optimised.predict(X_test)

accuracy = np.mean(preds == y_test)

print(f"{LR_Optimised.__class__.__name__} | Sample {n_sample} | Accuracy {accuracy:.5f}")