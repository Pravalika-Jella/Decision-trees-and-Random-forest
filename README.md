## Decision Tree & Random Forest Classification
## Overview
This repository demonstrates classification using Decision Tree and Random Forest algorithms with Python and scikit-learn.
By default, it uses the built-in sklearn breast cancer dataset and is designed for general tabular classification tasks.

## Requirements
bash
pip install pandas numpy matplotlib seaborn scikit-learn
## How to Use
Open the script (.ipynb or .py) in Jupyter Notebook, Google Colab, or your preferred IDE.

Default Usage:
Runs automatically with sklearn’s breast cancer dataset—no changes required.

Custom Dataset:

To use another CSV dataset, update the data loading section like so:

python
df = pd.read_csv("/path/to/your.csv")
target_col = 'target'
X = df.drop(columns=[target_col])
y = df[target_col]
Ensure your data contains a "target" column for classification.

## Run all cells.
The code will:

 Load and split your data

Train a Decision Tree and a Random Forest model

Print metrics: accuracy, precision, recall, F1-score, confusion matrix

Plot the top levels of the tree and top 10 feature importances

## Outputs
Metrics: Accuracy, precision, recall, F1-score, confusion matrix (printed)

Plots: Decision Tree visualization (top levels) and Random Forest feature importance bar chart

## Workflow Steps
Import libraries and load data

Split data into training and test sets

Train and evaluate Decision Tree

Visualize tree structure

Train and evaluate Random Forest

Print and plot feature importances
