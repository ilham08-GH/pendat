import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("GNB_model.pkl")
scaler = joblib.load("scaler_ecoli.pkl")

# Judul halaman
st.set_page_config(page_title="Prediksi Lokalisasi Protein E. coli", page_icon="🧬")
st.title("🧬 Prediksi Lokalisasi Protein E. coli")
st.markdown("Gunakan model Gaussian Naive Bayes untuk memprediksi lokasi subseluler dari protein E. coli.")

# Input fitur protein
st.subheader("📥 Masukkan Fitur Protein:")
mcg = st.number_input("mcg", value=0.0)
gvh = st.number_input("gvh", value=0.0)
lip = st.number_input("lip", value=0.0)
chg = st.number_input("chg", value=0.0)
aac = st.number_input("aac", value=0.0)
alm1 = st.number_input("alm1", value=0.0)
alm2 = st.number_input("alm2", value=0.0)

# Tombol prediksi
if st.button("🔍 Prediksi Lokasi"):
    fitur = np.array([[mcg, gvh, lip, chg, aac, alm1, alm2]])
    fitur_scaled = scaler.transform(fitur)  # normalisasi input
    hasil = model.predict(fitur_scaled)[0]
    st.success(f"✅ Prediksi Lokasi Protein: **{hasil.upper()}**")
