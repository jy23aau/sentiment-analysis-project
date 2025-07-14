# preprocess.py
import pandas as pd
import nltk
import spacy

nltk.download('stopwords')
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return " ".join(tokens)

df = pd.read_csv("data/raw.csv")
df['cleaned'] = df['teaching'].apply(preprocess)
df.to_csv("data/cleaned.csv", index=False)
