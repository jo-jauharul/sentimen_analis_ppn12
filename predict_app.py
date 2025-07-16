import joblib
import numpy as np
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Word2Vec
import re

# Load model dan vectorizer
model = joblib.load("model/naivebayes_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
w2v = Word2Vec.load("model/word2vec.model")

# Preprocessing tools
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words("indonesian"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def text_to_w2v(text, model):
    words = text.split()
    vector_size = model.vector_size
    vec = np.zeros(vector_size)
    count = 0
    for word in words:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    return vec, count  # Return count for validation

# Input dari user
input_text = input("Masukkan kalimat: ")
cleaned = clean_text(input_text)

# Validasi 1: Kosong setelah preprocessing
if not cleaned.strip():
    print("Kalimat tidak valid atau terlalu pendek setelah preprocessing.")
    exit()

# Validasi 2: Jumlah kata yang dikenali di Word2Vec terlalu sedikit
_, word2vec_known_count = text_to_w2v(cleaned, w2v)
if word2vec_known_count < 2:
    print("Kalimat terlalu asing atau tidak dikenali sistem. Silakan masukkan kalimat yang lebih relevan.")
    exit()

# Proses vektorisasi
tfidf_vec = tfidf.transform([cleaned]).toarray()
w2v_vec, _ = text_to_w2v(cleaned, w2v)
w2v_vec = w2v_vec.reshape(1, -1)

# Gabungkan fitur
features = np.hstack((tfidf_vec, w2v_vec))

# Prediksi
prediction = model.predict(features)
print("Hasil klasifikasi sentimen:", prediction[0])
