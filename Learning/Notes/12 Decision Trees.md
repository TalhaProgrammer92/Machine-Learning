---

## 🌳 What is a Decision Tree?

A **Decision Tree** mimics human decision-making. It splits the data into branches based on features and thresholds until it reaches a decision (leaf node).

* **Root Node**: The first feature split.
* **Internal Nodes**: Decision points (e.g., "Is age > 30?")
* **Leaves**: Final output (class or value)

---

## 🔍 Real-Life Examples

1. **Loan Approval (Classification)**

   * Input: Age, Salary, Credit Score
   * Output: Approve or Reject

2. **House Price Estimation (Regression)**

   * Input: Area, Location, Bedrooms
   * Output: Price (continuous value)

---

## 🐍 Decision Tree with `Scikit-learn` – Classification Example

### 🎯 Problem: Classify if a person will buy a computer based on age and income.

---

### 🧪 Step-by-Step Code

```python
# 1. Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. Sample dataset
import pandas as pd

data = {
    'Age': [25, 45, 35, 33, 22, 42],
    'Income': [50000, 100000, 60000, 120000, 40000, 90000],
    'BuysComputer': [0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# 3. Features & Target
X = df[['Age', 'Income']]
y = df['BuysComputer']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 5. Create and train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### 🔎 Visualizing the Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_tree(model, feature_names=['Age', 'Income'], class_names=['No', 'Yes'], filled=True)
plt.show()
```

---

## 📉 Regression Example: Predicting House Prices

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Dataset
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 250000, 300000, 350000, 400000])

# Train model
reg = DecisionTreeRegressor()
reg.fit(X, y)

# Predict
print(reg.predict([[2200]]))  # Estimate for 2200 sq.ft house
```

---

## ✅ Key Advantages of Decision Trees

* Easy to understand and visualize.
* Handles both numerical and categorical data.
* No need for feature scaling.
* Can model non-linear relationships.

---

## ⚠️ Limitations

* Prone to **overfitting** (can memorize training data).
* Sensitive to small changes in data (high variance).
* Not ideal for very large datasets (can become too complex).

🔁 Solution: Use **Random Forests** or **Gradient Boosted Trees** to overcome these issues.

---
