import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocess import clean_text

# Load datasets
fake_df = pd.read_csv("../dataset/Fake.csv")
real_df = pd.read_csv("../dataset/True.csv")

# Add labels
fake_df["label"] = 0
real_df["label"] = 1

# Merge and shuffle
df = pd.concat([fake_df, real_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

df = df[["text", "label"]]

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("../model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")
