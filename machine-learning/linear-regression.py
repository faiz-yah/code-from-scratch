import numpy as np 

# ─────────────────────────────────────────────
# Implementation 1 - Row by Row
# ─────────────────────────────────────────────

# Linear Regression requires 3 core steps:
# 1. Calculate cost function 
# 2. Calculate gradient
# 3. Calculate gradient descent


class LinearRegressionRowByRow():
    def __init__(self):
        pass
    
    def calculate_mse(self, X, y, w, b):
        m = len(X)
        total_error = 0
        
        for i in range(m):
            # For each sample of X, we do the prediction
            prediction = (w * X[i]) + b
            error = (prediction - y[i])**2
            # We sum up each prediction's error to a total error
            total_error += error
        
        # We obtain the final mse by dividing the quantity of samples
        mse = (1/(2*m)) * total_error

        return mse
    
    def calculate_gradient(self, X, y, w, b):
        m = len(X)
        
        dc_dw = 0
        dc_db = 0
        
        for i in range(m):
            prediction = (w * X[i]) + b
            dc_dw += (prediction - y[i]) * X[i]
            dc_db += (prediction - y[i])
        
        return (dc_dw/m), (dc_db/m)
    
    def gradient_descent(self, X, y, lr, n_iterations):
        m = len(X)
        w=0
        b=0
        
        for _ in range(n_iterations):
            dc_dw, dc_db = self.calculate_gradient(X, y, w, b)
            
            w -= (lr * dc_dw)
            b -= (lr * dc_db)
            
        return w, b
    

# ─────────────────────────────────────────────
# Implementation 2 - Vectorised
# ─────────────────────────────────────────────

class LinearRegressionVectorised():
    
    def __init__(self):
        pass
    
    def compute_mse(self, X, y, theta):
        m = len(X)
        predictions = X.dot(theta)
        errors = predictions - y
        mse = (1/ (2*m) ) * np.dot(errors, errors)
        
        return mse
    
    def gradient_descent(self, X, y, theta, lr, n_iterations):
        
        m = len(X)
        #cost_history = []
        
        for i in range(n_iterations):
            predictions = X.dot(theta)
            errors = predictions - y
            gradient = (1/m) * X.T.dot(errors)
            theta = theta - (lr * gradient)
            #cost_history.append(self.compute_mse(X, y, theta))
            
        return theta
    

# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "Salary Data.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "krishnaraj30/salary-prediction-data-simple-linear-regression",
  file_path
)

X_train = df['YearsExperience']
y_train = df['Salary']


LR_v1 = LinearRegressionRowByRow()

lr = 0.01
n_iterations = 10000

w, b = LR_v1.gradient_descent(X_train, y_train, lr, n_iterations)


print(f"{LR_v1.__class__.__name__} | Weight is {w:.2f}, Bias is {b:.2f}")


LR_v2 = LinearRegressionVectorised()

X_train = df['YearsExperience']
y_train = df['Salary']

X_train = X_train.values.reshape(-1, 1)

X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]

theta = np.zeros(X_train_b.shape[1]) 
lr = 0.01
n_iterations = 10000

theta = LR_v2.gradient_descent(X_train_b, y_train, theta, lr, n_iterations)

print(f"{LR_v2.__class__.__name__}| Weight is {theta[1]:.2f}, Bias is {theta[0]:.2f}")



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────
# Scikit-learn baseline
# ─────────────────────────────────────────────
X_sk = df['YearsExperience'].values.reshape(-1, 1)
y_sk = df['Salary'].values

sk_model = LinearRegression()
sk_model.fit(X_sk, y_sk)
print(f"{'Scikit-learn'} | Weight is {sk_model.coef_[0]:.2f}, Bias is {sk_model.intercept_:.2f}")