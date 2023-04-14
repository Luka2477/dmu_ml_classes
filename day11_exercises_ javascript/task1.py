from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from numpy import column_stack

df = pd.read_csv('data/StudentGrades.csv')

print(df.head())
print(df.tail())
print(df.describe())

plt.scatter(x=df.study_hours, y=df.student_marks)
plt.xlabel("Student Study Hours")
plt.ylabel("Student Marks")
plt.title("Scatter Plot of Student Study Hours vs Student Marks")
plt.show()

print(df.isnull().sum())

df2 = df.fillna(df.mean())
print(df2.isnull().sum())
print(df2.head())

X = df2.drop("student_marks", axis="columns")
y = df2.drop("study_hours", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
print("Shape of X_train = ", X_train.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of X_test = ", X_test.shape)
print("Shape of y_test = ", y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")

train_combined = column_stack((X_train, y_train))
kmeans.fit(train_combined)
train_labels = kmeans.labels_

test_combined = column_stack((X_test, y_test))
kmeans.fit(test_combined)
test_labels = kmeans.labels_

plt.scatter(X_train, y_train, c=train_labels)
plt.show()

plt.scatter(X_test, y_test, c=test_labels)
plt.plot(X_train, lr.predict(X_train), color="r")
plt.show()
