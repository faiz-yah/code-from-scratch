# ─────────────────────────────────────────────
# Implementation
# ─────────────────────────────────────────────

import numpy as np 
from collections import Counter

def calculate_distance(x1, x2):
    return np.sqrt(np.sum((x2-x1)**2))


class KNN():
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]    
        return np.array(y_pred)
        
    def _predict(self, x):
        """
        Takes in the dataset and does the prediction
        """
        
        # Calculate distance for every point
        distances = [calculate_distance(x, x_train) for x_train in self.X_train]
        
        # Obtain the top k closest points
        idx_closest = np.argsort(distances)[:self.k]
        knn_labels = [self.y_train[i] for i in idx_closest]
        
        # Return most common class
        most_common = Counter(knn_labels).most_common(1)
        return most_common[0][0]


# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

cls = KNN(k=3)

# Train
cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)

accuracy = round((np.sum(y_pred == y_test) / len(y_test)),2)

## Test out multiple k variations

result = []
for k in range(1,100):
    cls = KNN(k=k)

    # Train
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)

    accuracy = round((np.sum(y_pred == y_test) / len(y_test)),2)
    
    result.append({
        "K-neighbours": k,
        "Accuracy": accuracy
    })

    print(f"Current k neightbour is {k}")
    print(f"Accuracy is {accuracy} ")

result = pd.DataFrame(result)

print(result)

x = result['K-neighbours']
y = result['Accuracy']


plt.figure()
plt.plot(x, y)
plt.title("K-neighbours vs Accuracy")
plt.xlabel("K-neigbours")
plt.ylabel("Accuracy")
plt.show()