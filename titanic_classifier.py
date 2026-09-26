# ============================================================
# TITANIC SURVIVAL CLASSIFICATION PROJECT
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)

# OpenML provides the Titanic dataset
from sklearn.datasets import fetch_openml

titanic = fetch_openml(
    name="titanic",
    version=1,
    as_frame=True
)

df = titanic.frame

print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. INSPECT DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("2. DATA INSPECTION")
print("=" * 60)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe(include="all").T)


# ------------------------------------------------------------
# 3. CLEAN TARGET VARIABLE
# ------------------------------------------------------------

# Convert target variable into integer
df["survived"] = df["survived"].astype(int)

print("\nTarget distribution:")
print(df["survived"].value_counts())


# ------------------------------------------------------------
# 4. FEATURE SET 1 - BASIC FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("3. FEATURE SET 1 - BASIC FEATURES")
print("=" * 60)

# Basic features
basic_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X_basic = df[basic_features].copy()
y = df["survived"]


# ------------------------------------------------------------
# 5. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_basic,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ------------------------------------------------------------
# 7. MODEL 1 - LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. MODEL 1 - LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

print("\nLogistic Regression Accuracy:")
print(f"{logistic_accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_logistic
    )
)


# ------------------------------------------------------------
# 8. CONFUSION MATRIX - LOGISTIC REGRESSION
# ------------------------------------------------------------

cm_logistic = confusion_matrix(
    y_test,
    y_pred_logistic
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 9. MODEL 2 - RANDOM FOREST
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. MODEL 2 - RANDOM FOREST")
print("=" * 60)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

y_pred_rf = random_forest_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("\nRandom Forest Accuracy:")
print(f"{rf_accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)


# ------------------------------------------------------------
# 10. CONFUSION MATRIX - RANDOM FOREST
# ------------------------------------------------------------

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. MODEL COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ]
})

print("\n")
print(results)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.ylim(0, 1)

plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Model Accuracy Comparison")

plt.tight_layout()
plt.show()


# ============================================================
# STRETCH GOAL
# ============================================================

print("\n" + "=" * 60)
print("7. STRETCH GOAL - TITLE FEATURE")
print("=" * 60)


# ------------------------------------------------------------
# 12. EXTRACT TITLE FROM NAME
# ------------------------------------------------------------

def extract_title(name):

    if pd.isna(name):
        return "Unknown"

    # Example:
    # "Braund, Mr. Owen Harris"
    #                     ↑
    #                   title

    title = name.split(",")[1].split(".")[0].strip()

    return title


df["Title"] = df["name"].apply(extract_title)


print("\nExtracted titles:")
print(df["Title"].value_counts())


# ------------------------------------------------------------
# 13. GROUP RARE TITLES
# ------------------------------------------------------------

common_titles = [
    "Mr",
    "Miss",
    "Mrs",
    "Master"
]

df["Title"] = df["Title"].apply(
    lambda x: x if x in common_titles else "Rare"
)

print("\nTitles after grouping:")
print(df["Title"].value_counts())


# ------------------------------------------------------------
# 14. FEATURE SET 2
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("8. FEATURE SET 2 - WITH TITLE")
print("=" * 60)

features_with_title = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "Title"
]

X_title = df[features_with_title].copy()

y_title = df["survived"]


# ------------------------------------------------------------
# 15. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(
    X_title,
    y_title,
    test_size=0.2,
    random_state=42,
    stratify=y_title
)


# ------------------------------------------------------------
# 16. PREPROCESSING FOR TITLE MODEL
# ------------------------------------------------------------

numeric_features_title = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features_title = [
    "sex",
    "embarked",
    "Title"
]


numeric_transformer_title = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_transformer_title = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor_title = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer_title,
            numeric_features_title
        ),
        (
            "cat",
            categorical_transformer_title,
            categorical_features_title
        )
    ]
)


# ------------------------------------------------------------
# 17. RANDOM FOREST WITH TITLE
# ------------------------------------------------------------

title_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor_title
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


title_model.fit(
    X_train_title,
    y_train_title
)


y_pred_title = title_model.predict(
    X_test_title
)


title_accuracy = accuracy_score(
    y_test_title,
    y_pred_title
)


# ------------------------------------------------------------
# 18. STRETCH GOAL RESULTS
# ------------------------------------------------------------

print("\nRandom Forest + Title Accuracy:")
print(f"{title_accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test_title,
        y_pred_title
    )
)


# ------------------------------------------------------------
# 19. CONFUSION MATRIX
# ------------------------------------------------------------

cm_title = confusion_matrix(
    y_test_title,
    y_pred_title
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_title,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=[
        "Did Not Survive",
        "Survived"
    ],
    yticklabels=[
        "Did Not Survive",
        "Survived"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Random Forest + Title - Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. FINAL COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("9. FINAL COMPARISON")
print("=" * 60)

final_results = pd.DataFrame({
    "Experiment": [
        "Logistic Regression - Basic Features",
        "Random Forest - Basic Features",
        "Random Forest - Basic + Title"
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy,
        title_accuracy
    ]
})

print("\n")
print(final_results)


# ------------------------------------------------------------
# 21. ACCURACY IMPROVEMENT
# ------------------------------------------------------------

improvement = title_accuracy - rf_accuracy

print("\nAccuracy change after adding Title:")

if improvement > 0:
    print(
        f"Accuracy improved by "
        f"{improvement:.4f} "
        f"({improvement * 100:.2f} percentage points)"
    )

elif improvement < 0:
    print(
        f"Accuracy decreased by "
        f"{abs(improvement):.4f} "
        f"({abs(improvement) * 100:.2f} percentage points)"
    )

else:
    print("Accuracy remained unchanged.")


# ------------------------------------------------------------
# 22. FINAL GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=final_results,
    x="Accuracy",
    y="Experiment"
)

plt.xlim(0, 1)

plt.xlabel("Accuracy")
plt.ylabel("Experiment")

plt.title(
    "Titanic Model and Feature Comparison"
)

plt.tight_layout()

plt.show()


print("\n" + "=" * 60)
print("PROJECT COMPLETED!")
print("=" * 60)