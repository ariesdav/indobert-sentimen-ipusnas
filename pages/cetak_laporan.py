#cetak_laporan.py
import streamlit as st
from datetime import datetime
from utils.report_generator import (
    generate_pdf_data_ulasan,
    generate_pdf_preprocessing,
    generate_pdf_klasifikasi,
    generate_pdf_model,
    generate_excel,
)

st.title("📥 Ekspor Laporan")

# kalau belum ada hasil klasifikasi, stop di sini
if "classified_df" not in st.session_state:
    st.warning(
        "Belum ada hasil klasifikasi. Silakan upload data dan jalankan klasifikasi terlebih dahulu.",
        icon="⚠️"
    )
    st.stop()

df = st.session_state["classified_df"]
total = len(df)
count_positif = int((df["sentimen"] == "positive").sum())
count_negatif = int((df["sentimen"] == "negative").sum())
count_netral = int((df["sentimen"] == "neutral").sum())

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

st.write(
    "Unduh laporan analisis sentimen dalam 4 bagian terpisah, atau unduh "
    "rekap lengkap dalam format Excel."
)

jumlah_baris = st.selectbox(
    "Jumlah ulasan ditampilkan di tabel PDF (Laporan Data Ulasan & Hasil Klasifikasi)",
    options=[25, 50, 100],
    index=0,
    help="Berlaku untuk laporan yang menampilkan tabel data ulasan, agar PDF tetap ringkas dan mudah dibaca."
)

st.divider()

# Laporan Data Ulasan (Mentah)
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("1️⃣ Laporan Data Ulasan (Mentah)")
    st.caption("Ringkasan jumlah data & tabel ulasan mentah sebelum diproses.")
with col2:
    if st.button("Buat Laporan 1", use_container_width=True, key="btn_data_ulasan"):
        with st.spinner("Menyusun laporan data ulasan..."):
            buffer = generate_pdf_data_ulasan(df, max_rows=jumlah_baris)
            st.session_state["buf_data_ulasan"] = buffer
    if "buf_data_ulasan" in st.session_state:
        st.download_button(
            "📥 Unduh PDF",
            data=st.session_state["buf_data_ulasan"],
            file_name=f"laporan_data_ulasan_{timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_data_ulasan",
        )

st.divider()

#Laporan Preprocessing
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("2️⃣ Laporan Preprocessing")
    st.caption("Wordcloud sebelum vs sesudah, dan contoh tahapan preprocessing.")
with col2:
    if st.button("Buat Laporan 2", use_container_width=True, key="btn_preprocessing"):
        with st.spinner("Menyusun laporan preprocessing..."):
            buffer = generate_pdf_preprocessing(df)
            st.session_state["buf_preprocessing"] = buffer
    if "buf_preprocessing" in st.session_state:
        st.download_button(
            "📥 Unduh PDF",
            data=st.session_state["buf_preprocessing"],
            file_name=f"laporan_preprocessing_{timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_preprocessing",
        )

st.divider()

# Laporan Hasil Klasifikasi
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("3️⃣ Laporan Hasil Klasifikasi Sentimen")
    st.caption("Ringkasan statistik, proporsi sentimen, dan tabel hasil klasifikasi.")
with col2:
    if st.button("Buat Laporan 3", use_container_width=True, key="btn_klasifikasi"):
        with st.spinner("Menyusun laporan klasifikasi..."):
            buffer = generate_pdf_klasifikasi(df, max_rows=jumlah_baris)
            st.session_state["buf_klasifikasi"] = buffer
    if "buf_klasifikasi" in st.session_state:
        st.download_button(
            "📥 Unduh PDF",
            data=st.session_state["buf_klasifikasi"],
            file_name=f"laporan_klasifikasi_{timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_klasifikasi",
        )

st.divider()

#Laporan 4: Performa Model
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("4️⃣ Laporan Performa Model")
    st.caption("Confusion matrix, precision/recall/F1-score dari hasil training.")
with col2:
    if st.button("Buat Laporan 4", use_container_width=True, key="btn_model"):
        with st.spinner("Menyusun laporan performa model..."):
            buffer = generate_pdf_model()
            st.session_state["buf_model"] = buffer
    if "buf_model" in st.session_state:
        st.download_button(
            "📥 Unduh PDF",
            data=st.session_state["buf_model"],
            file_name=f"laporan_performa_model_{timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_model",
        )

st.divider()

#Rekap lengkap Excel
st.subheader("📊 Rekap Lengkap (Excel)")
st.caption("Ringkasan statistik + seluruh data ulasan beserta confidence per baris.")
if st.button("Buat Rekap Excel", key="btn_excel"):
    with st.spinner("Menyusun rekap Excel..."):
        buffer = generate_excel(df)
        st.session_state["buf_excel"] = buffer
if "buf_excel" in st.session_state:
    st.download_button(
        "📥 Unduh Excel",
        data=st.session_state["buf_excel"],
        file_name=f"laporan_sentimen_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_excel",
    )

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🏠 Kembali ke Beranda", use_container_width=True):
        st.switch_page("pages/beranda.py")
with col_b:
    if st.button("📤 Analisis Data Baru", use_container_width=True):
        st.switch_page("pages/hal_preprocessing.py")