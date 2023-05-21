import random
import sys
import ssl

from nltk import download
from nltk.corpus import twitter_samples

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


# Have to create certificate to download faces
ssl._create_default_https_context = ssl._create_unverified_context

# Download twitter samples dataset
download('twitter_samples')

pos_tweets = [(string, 1) for string in twitter_samples.strings('positive_tweets.json')]
neg_tweets = [(string, 0) for string in twitter_samples.strings('negative_tweets.json')]
pos_tweets.extend(neg_tweets)
comb_tweets = pos_tweets
random.shuffle(comb_tweets)
tweets, labels = (zip(*comb_tweets))

count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X = count_vectorizer.fit_transform(tweets)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=10)

# Exercise 1:
# Here we are using a Randomforest classifier. Using this technique (see the code) can you improve
# the accuracy on the test set?
print("Use Randomforest to classify")
rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=2)
rf.fit(X_train, y_train)
print()

rf_preds = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.6f}")
print(f"Confusion matrix: \n{confusion_matrix(y_test, rf_preds)}")
print()

test_text = ["Bad movie. Stinks. Terrible actors. Plot makes no sense."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, rf.predict(Example_Tweet.toarray()))

test_text = ["Great movie. Enjoyed it a lot. Wonderful actors. Best story ever."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, rf.predict(Example_Tweet.toarray()))
print()

# Exercise 2:
# We could also have used a neural net to classify.
# How? Make that work as well! What do think about the accuracy the neural net method gives?
print("Use Multi-Layer Percepton to classify")
mlp = MLPClassifier(solver="lbfgs", hidden_layer_sizes=[12, 8], max_iter=sys.maxsize)
mlp.fit(X_train, y_train)
print()

mlp_preds = mlp.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, mlp_preds):.6f}")
print(f"Confusion matrix: \n{confusion_matrix(y_test, mlp_preds)}")
print()

test_text = ["Bad movie. Stinks. Terrible actors. Plot makes no sense."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, mlp.predict(Example_Tweet.toarray()))

test_text = ["Great movie. Enjoyed it a lot. Wonderful actors. Best story ever."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, mlp.predict(Example_Tweet.toarray()))
print()

# Exercise 3:
# Test the two classifiers with texts you provide (good and bad sentiments). Does it work?
print("These should be classified as negative (0)")
test_text = ["This the worst song that I have ever heard in my life. How did they even let him record that???"]
Example_Tweet = count_vectorizer.transform(test_text)
print("Text          :", test_text[0])
print("Random Forest :", rf.predict(Example_Tweet.toarray()))
print("MLP           :", mlp.predict(Example_Tweet.toarray()))
print()

print("These should be classified as positive (1)")
test_text = ["Why is it that Marilyn Monroe was so good looking? "
             "Let's just all take a moment to appreciate the amazing things she did for us!"]
Example_Tweet = count_vectorizer.transform(test_text)
print("Text          :", test_text[0])
print("Random Forest :", rf.predict(Example_Tweet.toarray()))
print("MLP           :", mlp.predict(Example_Tweet.toarray()))
print()

print("These should be classified as positive (1)")
test_text = ["OMG I love Marilyn Monroe. She is just the best actor ever and she looks great too!"]
Example_Tweet = count_vectorizer.transform(test_text)
print("Text          :", test_text[0])
print("Random Forest :", rf.predict(Example_Tweet.toarray()))
print("MLP           :", mlp.predict(Example_Tweet.toarray()))
