import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

train_df = pd.read_csv('data/bank_loan_data.csv')
print(train_df.info())

# We can see there are total 13 columns including target variable.
train_df = train_df.drop(columns=['Loan_ID'])  # Dropping Loan ID
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']
print(categorical_columns)
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print(numerical_columns)

fig, axes = plt.subplots(4, 2, figsize=(12, 15))
for idx, cat_col in enumerate(categorical_columns):
    row, col = idx // 2, idx % 2
    sns.countplot(x=cat_col, data=train_df, hue='Loan_Status', ax=axes[row, col])

plt.subplots_adjust(hspace=1)
plt.show()

#  Pre-processing
#  Encoding Categorical Features.
#  Imputing missing values
train_df_encoded = pd.get_dummies(train_df, drop_first=True)
train_df_encoded.head()

# training and test Data
# Split Features and Target Varible
X = train_df_encoded.drop(columns='Loan_Status_Y')
y = train_df_encoded['Loan_Status_Y']

# Splitting into Train -Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Handling/Imputing Missing values
imp = SimpleImputer(strategy='mean')
imp_train = imp.fit(X_train)
X_train = imp_train.transform(X_train)
X_test = imp_train.transform(X_test)

logreg_train_accuracies = []
logreg_test_accuracies = []

randfor_train_accuracies = []
randfor_test_accuracies = []

mlp_train_accuracies = []
mlp_test_accuracies = []

thresholds = []

logreg_clf = LogisticRegression(solver='liblinear')
randfor_clf = RandomForestClassifier(max_depth=None, n_estimators=100)
mlp_clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 25, 10), max_iter=sys.maxsize)

for thresh in np.arange(0.1, 0.9, 0.1):  # Sweeping from threshold of 0.1 to 0.9
    # Logistic Regression
    logreg_clf.fit(X_train, y_train)

    logreg_y_pred_train_thresh = logreg_clf.predict_proba(X_train)[:, 1]
    logreg_y_pred_train = (logreg_y_pred_train_thresh > thresh).astype(int)

    logreg_train_acc = accuracy_score(y_train, logreg_y_pred_train)

    logreg_y_pred_test_thresh = logreg_clf.predict_proba(X_test)[:, 1]
    logreg_y_pred_test = (logreg_y_pred_test_thresh > thresh).astype(int)

    logreg_test_acc = accuracy_score(y_test, logreg_y_pred_test)

    logreg_train_accuracies.append(logreg_train_acc)

    logreg_test_accuracies.append(logreg_test_acc)

    # Random Forest
    randfor_clf.fit(X_train, y_train)

    randfor_y_pred_train_thresh = randfor_clf.predict_proba(X_train)[:, 1]
    randfor_y_pred_train = (randfor_y_pred_train_thresh > thresh).astype(int)

    randfor_train_acc = accuracy_score(y_train, randfor_y_pred_train)

    randfor_y_pred_test_thresh = randfor_clf.predict_proba(X_test)[:, 1]
    randfor_y_pred_test = (randfor_y_pred_test_thresh > thresh).astype(int)

    randfor_test_acc = accuracy_score(y_test, randfor_y_pred_test)

    randfor_train_accuracies.append(randfor_train_acc)

    randfor_test_accuracies.append(randfor_test_acc)

    # Multi-Layer Perceptron Classifier
    mlp_clf.fit(X_train, y_train)

    mlp_y_pred_train_thresh = mlp_clf.predict_proba(X_train)[:, 1]
    mlp_y_pred_train = (mlp_y_pred_train_thresh > thresh).astype(int)

    mlp_train_acc = accuracy_score(y_train, mlp_y_pred_train)

    mlp_y_pred_test_thresh = mlp_clf.predict_proba(X_test)[:, 1]
    mlp_y_pred_test = (mlp_y_pred_test_thresh > thresh).astype(int)

    mlp_test_acc = accuracy_score(y_test, mlp_y_pred_test)

    mlp_train_accuracies.append(mlp_train_acc)

    mlp_test_accuracies.append(mlp_test_acc)

    thresholds.append(thresh)

Threshold = {"Logreg Training Accuracy": logreg_train_accuracies, "Logreg Test Accuracy": logreg_test_accuracies,
             "Randfor Training Accuracy": randfor_train_accuracies, "MLP Training Accuracy": mlp_train_accuracies,
             "MLP Test Accuracy": mlp_test_accuracies, "Randfor Test Accuracy": randfor_test_accuracies,
             "Decision Threshold": thresholds}
Threshold_df = pd.DataFrame.from_dict(Threshold)

logreg_plot_df = Threshold_df.melt('Decision Threshold', var_name='Metrics', value_name="Values")
fig, ax = plt.subplots(figsize=(15, 8))
sns.pointplot(x="Decision Threshold", y="Values", hue="Metrics", data=logreg_plot_df, ax=ax)

plt.show()
