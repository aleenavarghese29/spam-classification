import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example training data
X_train = ["Hello", "Spam message", "Good morning", "Buy now"]
y_train = [0, 1, 0, 1]

# Create and fit the vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save both the vectorizer and model together in a tuple
with open("spam_classifier_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)