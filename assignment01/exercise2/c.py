from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from pandas import read_csv, DataFrame

train_size = 0.8
total_entries: int

# skipping the header
data = read_csv('data/titanic_800.csv', sep=',', header=0)
total_entries = data["Age"].size

# Replace unknown Pclass values with the mean Pclass value
# Do the same for Age, Cabin and Embarked
data["Pclass"].fillna(data["Pclass"].mean(), inplace=True)
data["Age"].fillna(data["Age"].mean(), inplace=True)
data["Cabin"].fillna("0", inplace=True)
data["Embarked"].fillna("S", inplace=True)

# Replace male and female sexes with integer values
data['Sex'] = data['Sex'].replace(['male'], 0)
data['Sex'] = data['Sex'].replace(['female'], 1)

# Convert cabin features to integer values treating them as hex
data["Cabin"] = data["Cabin"].apply(lambda e: sum([ord(c) for c in e]))

# Replace embarked values with integer representations
data['Embarked'] = data['Embarked'].replace(['S'], 0)
data['Embarked'] = data['Embarked'].replace(['Q'], 1)
data['Embarked'] = data['Embarked'].replace(['C'], 2)

# Copy the correct results
y_values = DataFrame(dict(Survived=[]), dtype=int)
y_values["Survived"] = data["Survived"].copy()

# Delete the y_values and unneeded features from the dataset
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Survived', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
data.drop('Fare', axis=1, inplace=True)

# Show the data
print("Description:\n", data.describe(include='all'))

# Make train and test datasets
X_train, X_test, y_train, y_test = train_test_split(data, y_values, train_size=train_size, stratify=y_values, )

# Scale data set
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create NN MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1000)
# The reason for the values.ravel is that these data has not been scaled and they need to be converted to the correct
# input format for the mlp.fit. Data that is scaled already has this done to them.
mlp.fit(X_train, y_train.values.ravel())

# Calculations
predictions = mlp.predict(X_test)
matrix = confusion_matrix(y_test, predictions)
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", matrix)
print(f"Percent right: {(matrix[0][0] + matrix[1][1]) / (total_entries * (1 - train_size))}%")
