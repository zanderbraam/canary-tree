import os
import json
import optuna

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score, \
    ConfusionMatrixDisplay

# Get the data
input_data = pd.read_csv("./data/transformed/input_data.csv")
target_data = pd.read_csv("./data/transformed/target_data.csv")

# Dropping the first unnamed column and merging datasets on the date column
input_data = input_data.drop(columns=input_data.columns[0])
target_data = target_data.drop(columns=target_data.columns[0])

# Merging the datasets
data = input_data.join(target_data)

# Separate the features and target variable
X = data.drop("strategy", axis=1)
y = data["strategy"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if os.path.exists("./data/best_parameters/best_hyperparameters.json"):
    # Load from JSON
    with open("./data/best_parameters/best_hyperparameters.json", "r") as f:
        loaded_params = json.load(f)

    model = XGBClassifier(**loaded_params)
    model.fit(X_train, y_train)

else:

    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True)
        }

        model = XGBClassifier(**param)
        model.set_params(early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        return accuracy


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Get best parameters
    best_params = study.best_trial.params

    # Save to JSON
    with open("./data/best_parameters/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f)

    # Retrain model with best parameters
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

# Plot feature importance
plot_importance(model, importance_type="gain", max_num_features=10, show_values=False)
plt.show()

# Calculate the probabilities of each class
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)

# Calculate average precision
average_precision = average_precision_score(y_test, y_probs)

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, label='Precision-recall curve (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.show()

# Plot confusion matrix
y_pred = model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion Matrix')
plt.show()

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1,
                                                        train_sizes=np.linspace(.1, 1.0, 5))

# Calculate mean and standard deviation for train/test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.title('Learning Curves')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc="best")
plt.show()

import pdb; pdb.set_trace()
