# --- Import library utama ---
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns

# --- Download stopwords bahasa Indonesia jika belum ada ---
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# --- Fungsi utama untuk preprocessing teks ---
def preprocess(text):
    if not isinstance(text, str):
        text = str(text) if not pd.isna(text) else ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)          # hapus link
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # hapus karakter selain huruf & spasi
    text = ' '.join([w for w in text.split() if w not in stop_words])  # hapus stopwords
    return text

# --- Custom CSS untuk mempercantik tampilan ---
st.markdown("""
    <style>
    .main {background-color: #f9fafb;}
    .sidebar .sidebar-content {background-color: #ffebc6;}
    .css-18e3th9 {background-color: #fff2cc;}
    .stButton>button {color: white; background: #e67e22;}
    .stDownloadButton>button {background: #16a085;}
    .stProgress > div > div > div > div {background-color: #e67e22;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigasi Utama ---
st.sidebar.image("https://img.icons8.com/color/96/000000/twitter--v1.png", width=60)
st.sidebar.title("Analisis Sentimen Twitter")
menu = st.sidebar.radio("Menu", [
    "Upload Data", "Preprocessing", "Training & Fitur", "Evaluasi", "Prediksi"
])
st.sidebar.markdown("""
---
*By: Muhammad Jauharul Maknun*  
Optimasi Naive Bayes, TF-IDF, Word2Vec
""")

# --- STATE: Menyimpan data pada sesi agar tidak hilang jika pindah menu ---
if 'df' not in st.session_state:
    st.session_state.df = None            # dataframe utama
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None    # vectorizer TF-IDF
if 'w2v_model' not in st.session_state:
    st.session_state.w2v_model = None     # model Word2Vec
if 'model' not in st.session_state:
    st.session_state.model = None         # model Naive Bayes
if 'fitur' not in st.session_state:
    st.session_state.fitur = 'TF-IDF'     # jenis fitur yg dipakai
if 'text_col' not in st.session_state:
    st.session_state.text_col = 'full_text'
if 'label_col' not in st.session_state:
    st.session_state.label_col = 'sentiment'

# --- MENU 1: UPLOAD DATASET CSV ---
if menu == "Upload Data":
    st.header("📂 1. Upload Data Tweet")

    # Jika sudah pernah upload, tampilkan data lama
    if st.session_state.df is not None:
        st.success("Data sudah diupload sebelumnya!")
        st.write(st.session_state.df.head(10))
        st.write("Kolom yang ditemukan:", list(st.session_state.df.columns))
        st.info("Lanjut ke menu Preprocessing setelah upload.")
    else:
        # Upload file CSV
        data_file = st.file_uploader("Upload file CSV (minimal ada kolom full_text & sentiment)", type=['csv'])
        if data_file:
            df = pd.read_csv(data_file)
            st.session_state.df = df        # simpan ke state
            st.success(f"File berhasil diupload! Jumlah data: {len(df)}")
            st.write(df.head(10))
            st.write("Kolom yang ditemukan:", list(df.columns))
            st.info("Lanjut ke menu Preprocessing setelah upload.")

# --- MENU 2: PREPROCESSING TEKS ---
elif menu == "Preprocessing":
    st.header("🧹 2. Preprocessing Data")
    df = st.session_state.df

    if df is not None:
        # Pilih kolom teks dan label (jika tidak default)
        text_col = 'full_text' if 'full_text' in df.columns else st.selectbox("Pilih kolom teks:", df.columns)
        label_col = 'sentiment' if 'sentiment' in df.columns else st.selectbox("Pilih kolom label:", df.columns)
        st.session_state.text_col = text_col
        st.session_state.label_col = label_col

        # Proses preprocessing
        with st.spinner("Melakukan preprocessing..."):
            df['clean'] = df[text_col].apply(preprocess)
        st.write("Hasil Preprocessing (semua data, bisa discroll):")
        st.dataframe(df[[text_col, 'clean', label_col]])

        # Tombol download hasil preprocessing
        csv = df[[text_col, 'clean', label_col]].to_csv(index=False)
        st.download_button("⬇ Download data hasil preprocessing", data=csv, file_name="data_clean.csv", mime="text/csv")

        st.info("Klik menu Training & Fitur setelah data bersih.")
    else:
        st.warning("Silakan upload data terlebih dahulu.")

# --- MENU 3: PILIH FITUR & TRAINING MODEL ---
elif menu == "Training & Fitur":
    st.header("⚡ 3. Pilih Fitur & Training Model")
    df = st.session_state.df

    if df is not None and 'clean' in df:
        # Pilih fitur
        fitur = st.radio("Pilih fitur:", ["TF-IDF", "Word2Vec"])
        st.session_state.fitur = fitur
        X, model, vectorizer, w2v_model = None, None, None, None
        y = df[st.session_state.label_col]

        progress = st.progress(0)
        if fitur == "TF-IDF":
            # TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df['clean'])
            model = MultinomialNB()
            st.session_state.vectorizer = vectorizer
        else:
            # Word2Vec model
            w2v_model = Word2Vec([t.split() for t in df['clean']], vector_size=100, min_count=1)
            def get_w2v(text):
                words = text.split()
                word_vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
                return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(w2v_model.vector_size)
            X = np.array([get_w2v(t) for t in df['clean']])
            model = GaussianNB()
            st.session_state.w2v_model = w2v_model
        progress.progress(40)

        # Split data dan training model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.session_state.model = model
        progress.progress(100)

        st.success("Training selesai! Lanjut ke menu Evaluasi untuk melihat performa model.")
        st.write("Contoh hasil prediksi pada data uji:")
        st.dataframe(pd.DataFrame({ 'y_test': y_test[:10], 'y_pred': y_pred[:10] }))
    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")

# --- MENU 4: EVALUASI MODEL ---
elif menu == "Evaluasi":
    st.header("📊 4. Evaluasi Model")
    df = st.session_state.df

    if df is not None and st.session_state.model is not None:
        fitur = st.session_state.fitur
        vectorizer = st.session_state.vectorizer
        w2v_model = st.session_state.w2v_model
        model = st.session_state.model
        y = df[st.session_state.label_col]

        # Ekstrak fitur sesuai pilihan
        if fitur == "TF-IDF":
            X = vectorizer.transform(df['clean'])
        else:
            def get_w2v(text):
                words = text.split()
                word_vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
                return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(w2v_model.vector_size)
            X = np.array([get_w2v(t) for t in df['clean']])

        # Evaluasi pada data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        st.write("*Akurasi:* {:.2f}%".format(100*accuracy_score(y_test, y_pred)))
        st.write("*Classification Report:*")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

        # Visualisasi Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(list(set(y)))
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="YlOrBr")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)
    else:
        st.warning("Silakan lakukan training model terlebih dahulu.")

# --- MENU 5: PREDIKSI TWEET BARU ---
elif menu == "Prediksi":
    st.header("🔮 5. Prediksi Sentimen Tweet Baru")
    model = st.session_state.model
    vectorizer = st.session_state.vectorizer
    w2v_model = st.session_state.w2v_model
    fitur = st.session_state.fitur

    if model is not None:
        tweet = st.text_area("Masukkan tweet untuk diprediksi:")
        if st.button("Prediksi"):
            clean = preprocess(tweet)
            if fitur == "TF-IDF":
                feat = vectorizer.transform([clean])
            else:
                def get_w2v(text):
                    words = text.split()
                    word_vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
                    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(w2v_model.vector_size)
                feat = get_w2v(clean).reshape(1, -1)
            pred = model.predict(feat)[0]
            if pred == "positif":
                st.success(f"Sentimen: {pred} 😊")
            elif pred == "negatif":
                st.error(f"Sentimen: {pred} 😡")
            else:
                st.info(f"Sentimen: {pred}")
    else:
        st.warning("Silakan lakukan training model terlebih dahulu.")

# --- FOOTER ---
st.markdown("""
---
<center><small>Dikembangkan untuk tugas optimasi analisis sentimen PPN 12%<br>by <b>Muhammad Jauharul Maknun</b> | 2025</small></center>
""", unsafe_allow_html=True)