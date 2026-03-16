# ─────────────────────────────────────────────
# Implementation
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
from kagglehub import KaggleDatasetAdapter

from pathlib import Path


class NaiveBayes:
    
    def fit(self, X_train, y_train):
        #1. Store unique classes
        #2. Compute prior probability (just the count of Y)
        #3. Calculate mean and std (to use for likelihood later)
        
        # Store all the unique classes
        self.classes = np.unique(y_train)
        
        # Value count each class then divide by total -> [0.628, 0.372]
        self.priors = [len(y_train[y_train == c])/ len(y_train) for c in self.classes]
        
        # Calculate feature mean for each class (for usage towards Likelihood of Continuous variable's Gaussion Probability Density)
        self.means = [X_train[y_train==c].mean() for c in self.classes]
        self.stds = [X_train[y_train==c].std() for c in self.classes]
        
    def compute_likelihood(self, row, class_idx):
        
        # Initialise likelihood equals to 1
        likelihood = 1
        
        # Access each feature row by row, to calculate the likelihood for each row
        for feature in row.index:
            mean = self.means[class_idx][feature]
            std = self.stds[class_idx][feature]
            likelihood *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp((-(row[feature] - mean)**2) / (2 * std**2))
            
        return likelihood
    
    def predict(self, X):
        y_pred = []
        
        # For each of the row
        for _, row in X.iterrows():
            posteriors = []
            
            # For each of the 2 class of each row
            for i in range(len(self.classes)):
                # Calculate likelihood of each row
                likelihood = self.compute_likelihood(row, i)
                # Append both classes' posterior for each row
                posteriors.append(likelihood * self.priors[i])
            
            # Select posterior with the highest probability as the prediction class (amongst the two)
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)


# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "uciml/breast-cancer-wisconsin-data",
  "data.csv"
)

print("Succesfully loaded data from Kaggle!")
print(df.head())

X = df.drop(columns=['diagnosis', 'id', 'Unnamed: 32'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = NaiveBayes()

nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = np.mean(y_pred == y_test) *100

print(f"Accuracy: {accuracy:.2f}%")

