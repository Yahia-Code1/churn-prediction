# Customer Churn Prediction

This project builds a machine learning model to predict customer subscription churn using the Telco Customer Churn dataset.

## Problem
Telecommunication companies lose revenue when customers cancel their subscriptions. Predicting churn allows companies to intervene and retain customers.

## Dataset
Telco Customer Churn dataset (~7,000 customers).

Features include:
- contract type
- tenure
- monthly charges
- internet service
- payment method

Target variable:
Churn (Yes / No).

## Pipeline

1. Data preprocessing
   - handled missing values
   - encoded categorical variables
   - standardized numerical features

2. Model training
   - Random Forest classifier
   - class imbalance handled using `class_weight="balanced"`

3. Evaluation
   - F1 Score
   - ROC-AUC
   - confusion matrix
   - ROC curve

## Results

Validation metrics:

F1 Score: ~0.60  
ROC-AUC: ~0.83

Confusion matrix shows the model misses some churn cases due to class imbalance. This was partially mitigated using class_weight="balanced".

## Technologies

- Python
- scikit-learn
- pandas
- numpy
- matplotlib## Experiment Results 

## How to Run

Install dependencies:

pip install -r requirements.txt

Train the model: python src/train.py

Evaluate the model: python src/evaluate.py

## Mathematical Objective

The Random Forest classifier builds decision trees by minimizing **Gini impurity** at each split.

Gini impurity is defined as:

G = 1 − Σ (p_i)^2

Where:

- p_i is the probability of class i at a node.

The model selects splits that minimize impurity, resulting in more homogeneous nodes.

To address class imbalance in the churn dataset, the model was trained with:

class_weight="balanced"

which increases the penalty for misclassifying churn cases.
