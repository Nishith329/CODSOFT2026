import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Loading dataset...")

plots = []
genres = []

with open("train_data.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":::")

        if len(parts) >= 4:
            genre = parts[2].strip()
            plot = parts[3].strip()

            plots.append(plot)
            genres.append(genre)

df = pd.DataFrame({"plot": plots, "genre": genres})

print("Samples:", len(df))
print("Genres count:")
print(df["genre"].value_counts().head(10))

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
X_vec = vectorizer.fit_transform(df["plot"])

print("Training model...")
model = MultinomialNB()
model.fit(X_vec, df["genre"])

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("✅ Model trained correctly")
