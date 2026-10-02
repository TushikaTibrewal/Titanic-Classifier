# Titanic Survival Classification

A machine learning classification project that predicts whether a passenger survived the Titanic disaster.

## Objective

The objective of this project is to:

- Load and inspect the Titanic dataset
- Handle missing values
- Encode categorical variables
- Split the dataset into training and testing sets
- Train multiple classification models
- Evaluate model performance
- Compare different models and feature sets
- Engineer a new Title feature from passenger names

## Dataset

The project uses the Titanic dataset available through OpenML.

The target variable is:

`survived`

where:

- 0 = Did not survive
- 1 = Survived

## Features

The initial feature set contains:

- `pclass` - Passenger class
- `sex` - Passenger gender
- `age` - Passenger age
- `sibsp` - Number of siblings/spouses aboard
- `parch` - Number of parents/children aboard
- `fare` - Ticket fare
- `embarked` - Port of embarkation

## Data Preprocessing

The following preprocessing steps were performed:

1. Missing numerical values were replaced using the median.
2. Missing categorical values were replaced using the most frequent value.
3. Categorical variables were encoded using One-Hot Encoding.
4. Numerical features were standardized for Logistic Regression.
5. The dataset was divided into training and testing sets using an 80/20 split.

## Models

Two machine learning models were initially tested:

### 1. Logistic Regression

Logistic Regression was used as a simple baseline classification model.

### 2. Random Forest

Random Forest was used as a tree-based ensemble model.

The accuracy of both models was compared.

## Feature Engineering

As a stretch goal, a new feature called `Title` was extracted from the passenger's name.

Examples:

- Mr
- Mrs
- Miss
- Master

Rare titles were grouped into a single `Rare` category.

The Random Forest model was then trained again using this additional feature.

## Evaluation

The models were evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

The results are printed by the Python script and visualized using bar charts and confusion matrices.

## Experiments

The project compares:

1. Logistic Regression + Basic Features
2. Random Forest + Basic Features
3. Random Forest + Basic Features + Title

## Conclusion

The final results show how model choice and feature engineering affect classification performance.

The Title feature provides additional information about the passenger and can be evaluated to determine whether it improves the model's accuracy.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
