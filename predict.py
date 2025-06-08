import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Set judul
st.title("🚆 Analisis Sentimen Ulasan Pengguna KAI Access")

# Load tokenizer dan model
@st.cache_resource
def load_model_tokenizer():
    model = tf.keras.models.load_model("gru_word2vec_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_tokenizer()
maxlen = 100  # Panjang input seperti saat training

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi preprocessing
def cleansing(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text(text, slang_dict):
    words = text.split()
    normalized = [slang_dict.get(w, w) for w in words]
    return " ".join(normalized)

def preprocess_input(text, slang_dict):
    text = cleansing(text)
    text = normalize_text(text, slang_dict)
    text = stemmer.stem(text)
    return text

# Load slangword dictionary (pastikan tersedia jika digunakan)
try:
    with open("kamus_slangwords.pkl", "rb") as f:
        slang_dict = pickle.load(f)
except:
    slang_dict = {}

# Prediksi sentimen
def predict_sentiment(text):
    clean_text = preprocess_input(text, slang_dict)
    seq = tokenizer.texts_to_sequences([clean_text])
    pad = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(pad)[0]
    label = np.argmax(pred)
    confidence = float(np.max(pred))
    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return label_map[label], confidence

# UI Input
user_input = st.text_area("Masukkan ulasan Anda di sini:", height=150)

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip():
        label, conf = predict_sentiment(user_input)
        st.success(f"Sentimen terdeteksi: **{label}** (Kepercayaan: {conf:.2%})")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")

