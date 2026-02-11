import pickle
from preprocess import clean_text

# Load model
with open("../model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("../model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

news = input("Enter news text: ")

cleaned = clean_text(news)
vectorized = vectorizer.transform([cleaned])

prediction = model.predict(vectorized)

if prediction[0] == 0:
    print("Prediction: FAKE NEWS")
else:
    print("Prediction: REAL NEWS")
