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

## Technologies

- Python
- scikit-learn
- pandas
- numpy
- matplotlib