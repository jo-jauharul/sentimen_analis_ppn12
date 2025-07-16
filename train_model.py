import os
import pandas as pd
import re
import string
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from scipy.sparse import hstack

# Pastikan folder model/ tersedia
os.makedirs("model", exist_ok=True)

# Load dataset (pastikan kolom: full_text, sentiment)
df = pd.read_csv("data/hasil_klasifikasi.csv")

# Validasi kolom
if 'full_text' not in df.columns or 'sentiment' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'full_text' dan 'sentiment'")

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = stemmer.stem(text)
    return text

print("Melakukan preprocessing...")
df['text'] = df['full_text'].astype(str).apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Simpan TF-IDF
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

# Word2Vec training
print("Melatih Word2Vec...")
X_train_tok = [text.split() for text in X_train]
w2v = Word2Vec(sentences=X_train_tok, vector_size=100, window=5, min_count=1, workers=4)
w2v.save("model/word2vec.model")

# Fungsi konversi teks ke vektor Word2Vec
def text_to_w2v_vector(text, model):
    words = text.split()
    vec = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    return vec / count if count > 0 else vec

X_train_w2v = np.array([text_to_w2v_vector(text, w2v) for text in X_train])
X_test_w2v = np.array([text_to_w2v_vector(text, w2v) for text in X_test])

# Gabungkan TF-IDF (sparse) + Word2Vec (dense)
X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_w2v])
X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_w2v])
# Latih model Naive Bayes
print("Melatih model Naive Bayes...")
model = GaussianNB()
model.fit(X_train_combined, y_train)

# Evaluasi
y_pred = model.predict(X_test_combined)
print("\nEvaluasi:")
print(classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))

# Simpan model
joblib.dump(model, "model/naivebayes_model.pkl")
print("Model dan vectorizer berhasil disimpan.")
