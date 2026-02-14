import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Loading dataset...")

# ⭐ Important: latin-1 encoding
data = pd.read_csv("spam.csv", encoding="latin-1")

# ⭐ Keep only first two columns
data = data[["v1","v2"]]

# Rename columns
data.columns = ["label","message"]

X = data["message"]
y = data["label"]

print("Vectorizing text...")

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

print("Training model...")

model = MultinomialNB()
model.fit(X_vec, y)

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("✅ Spam Model Trained Successfully")
