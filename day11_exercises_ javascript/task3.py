from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import ssl

# Have to create certificate to download faces
ssl._create_default_https_context = ssl._create_unverified_context

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Did we get what we wanted?
print(twenty_train.target_names)
print(len(twenty_train.data))
print()

# Lets see what we have :
print("\n".join(twenty_train.data[0].split("\n")[:2]))
print(twenty_train.target_names[twenty_train.target[0]])
print()
print("\n".join(twenty_train.data[1000].split("\n")[:2]))
print(twenty_train.target_names[twenty_train.target[1000]])
print()
print("\n".join(twenty_train.data[1].split("\n")[:2]))
print(twenty_train.target_names[twenty_train.target[1]])
print()

# Punctuation and single letter words will be automatically removed.
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

# Occurrence count is a good start but there is an issue:
# longer documents will have higher average count values than shorter documents,
# even though they might talk about the same topics.

# To avoid these potential discrepancies it suffices to divide the number
# of occurrences of each word in a document by the total number of words in the document:
# these new features are called “tf” for Term Frequencies.

# Another refinement on top of tf is to downscale weights for words
# that occur in many documents in the corpus and are therefore less
# informative than those that occur only in a smaller portion of the corpus.

# This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 10, 10), random_state=1)
clf.fit(X_train_tfidf, twenty_train.target)

docs_new = ['Biblical texts are old', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
print()

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    # ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 50, 10), random_state=1)),
    ('clf', SVC(C=100)),
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)

print(np.mean(predicted == twenty_test.target))
