#Titanic dataset predictions

#import panda library and a few others we will need.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
# skipping the header
data = pd.read_csv('data/titanic_train_500_age_passengerclass.csv', sep=',', header=0)

# show the data
print(data.describe(include='all'))
#the describe is a great way to get an overview of the data
print(data.values)

# Replace unknown values. Unknown class set to 3
data["Pclass"].fillna(3, inplace=True)


# Replace unknown values. Unknown age set to 25
avg = data["Age"].mean() #set it to avg instead
data["Age"].fillna(avg, inplace=True)

# Replace unknown values. Unknown survival set to survived
data["Survived"].fillna(1, inplace=True)


yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
yvalues["Survived"] = data["Survived"].copy()
#now the yvalues should contain just the survived column

x = data["Age"]
y = data["Pclass"]
plt.figure()
plt.scatter(x.values, y.values, color='green', s=20)
plt.show()

#now we can delete the survived column from the data (because
#we have copied that already into the yvalues.
data.drop('Survived', axis=1, inplace=True)

data.drop('PassengerId', axis=1, inplace=True)

# show the data
print(data.describe(include='all'))

xtrain = data.head(400)
xtest = data.tail(100)

ytrain = yvalues.head(400)
ytest = yvalues.tail(100)

print(ytrain)

#scaler til data for the memes
scaler = StandardScaler()
scaler.fit(xtrain)

xtrain = scaler.transform(xtrain) # uden skalering er accuracy faldet med 5%
xtest = scaler.transform(xtest)

#batch size 5 = 0,73, batch size 10 = 0,71
#learning rate init default = 0.001
mlp = MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1000, batch_size='auto')
mlp.fit(xtrain, ytrain)

predictions = mlp.predict(xtest)
matrix = confusion_matrix(ytest, predictions)
print(matrix)

tn, fp, fn, tp = matrix.ravel()

accuracy = (tp + tn) / (tp+fp+fn+tn)
print("Accuracy: " + str(accuracy))

print(classification_report(ytest, predictions))