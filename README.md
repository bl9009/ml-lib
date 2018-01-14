# Python Machine Learning Library
by Benjamin Labas

This Python library contains custom implementations of machine learning models and algorithms, helping me to get a better understanding of how these algorithms work and to improve my intuition on how to make use of them.

Implemented in Python 3.6 and NumPy 1.13.

## Models
Following machine learning models are available so far:

### Supervised

#### Regressors
- Linear regression (Stochastic Gradient Descent, Normal Equation)
- Ridge (l2) and LASSO (l1) regression (included in Linear regression)
- Logistic regression
- Polynomial regression

#### Classifiers
- Logistic classification
- Decision Tree classification
- Support Vector Machines (under construction)
- Artificial Neural Networks
  - Feed Forward NN (under construction)

### Unsupervised

#### Clustering
- K-Means (Lloyd's algorithm)


## Others
Following algorithms/utilities are included:
- Cross-validation
- Multiclass classification (OneVsAll)
- Feature scaler (Min-max scaler, Standardizer)
- Metrics (RSS, MSE, LogLoss, Gini impurity etc.)

## Planned
Following features are planned:
- k-Nearest Neighbors classification
- NaiveBayes classification
- Ensemble learning methods (bagging, stacking)
- Boosting (AdaBoost)
- More Artificial Neural Networks:
  - Recurrent Neural Network
  - Convolutional Neural Network
