#beranda.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from inference import load_model
from utils.branding import LOGO_SVG

RESULTS_PATH = "data/last_classified_result.csv"

@st.cache_resource
def get_model_resources():
    return load_model()

# Animasi loading custom
loading_placeholder = st.empty()

# load model saat pertama kali aja
if "resources" not in st.session_state:
    with loading_placeholder.container():
        st.markdown(
            """
            <style>
            .loader-wrap {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 55vh;
            }
            .loader-icon {
                font-size: 52px;
                animation: bounce 1.1s ease-in-out infinite;
            }
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-14px); }
            }
            .loader-text {
                margin-top: 20px;
                color: #ddd;
                font-size: 15px;
                letter-spacing: 0.3px;
                font-family: sans-serif;
            }
            .loader-bar {
                margin-top: 16px;
                width: 220px;
                height: 6px;
                border-radius: 999px;
                background: #333;
                overflow: hidden;
            }
            .loader-bar-fill {
                height: 100%;
                width: 40%;
                border-radius: 999px;
                background: linear-gradient(90deg, #2f9e44, #f0ad4e, #e34948);
                animation: slide 1.2s ease-in-out infinite;
            }
            @keyframes slide {
                0% { margin-left: -40%; }
                100% { margin-left: 100%; }
            }
            </style>
            <div class="loader-wrap">
                <div class="loader-icon">📖</div>
                <div class="loader-text">Memuat model IndoBERT, mohon tunggu sebentar...</div>
                <div class="loader-bar"><div class="loader-bar-fill"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["resources"] = get_model_resources()

    loading_placeholder.empty()

# cek data lama
if "classified_df" not in st.session_state and os.path.exists(RESULTS_PATH):
    st.session_state["classified_df"] = pd.read_csv(RESULTS_PATH)

# Header dengan logo
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="width:48px; height:48px; flex-shrink:0;">{LOGO_SVG}</div>
        <h1 style="margin:0;">Sistem Analisis Sentimen Ulasan Aplikasi</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Analisis Sentimen terhadap Ulasan Aplikasi iPusnas di Google Play Store dengan Metode IndoBERT")

#box deskripsi
st.markdown(
    """
    <div style="background-color: #2b2b2b; border-left: 4px solid #888; 
                padding: 14px 18px; border-radius: 6px; margin: 10px 0;">
        <p style="margin: 0; color: #ddd; font-size: 14px; line-height: 1.6;">
            📖 Website ini digunakan untuk menganalisis sentimen pengguna terhadap aplikasi
            <b>iPusnas</b> berdasarkan ulasan di <b>Google Play Store</b> menggunakan model
            <b>IndoBERT</b>, guna mengklasifikasikan ulasan menjadi <b>positif</b>, <b>netral</b>,
            dan <b>negatif</b> secara otomatis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

def get_sample_reviews(df: pd.DataFrame, per_class: int = 2) -> pd.DataFrame:
    """
    Ambil beberapa ulasan terbaru dari masing-masing kelas sentimen.
    Diurutkan lagi berdasarkan urutan asli agar tetap berasa "terbaru".
    """
    samples = []
    for label in ["positive", "negative", "neutral"]:
        subset = df[df["sentimen"] == label]
        if len(subset) > 0:
            samples.append(subset.tail(per_class))

    if not samples:
        return df.tail(0)

    combined = pd.concat(samples)
    #urutkan kembali berdasar index asli
    return combined.sort_index(ascending=False)

# ringkasan hasil analisis
if "classified_df" in st.session_state:
    df = st.session_state["classified_df"]

    total = len(df)
    count_positif = int((df["sentimen"] == "positive").sum())
    count_negatif = int((df["sentimen"] == "negative").sum())
    count_netral = int((df["sentimen"] == "neutral").sum())

    st.write(f"Menampilkan ringkasan dari **{total:,} ulasan** yang telah dianalisis.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ulasan", f"{total:,}")
    c2.metric("😊 Positif", f"{count_positif:,}", f"{count_positif/total*100:.1f}%")
    c3.metric("😠 Negatif", f"{count_negatif:,}", f"-{count_negatif/total*100:.1f}%", delta_color="inverse")
    c4.metric("😐 Netral", f"{count_netral:,}")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Proporsi Sentimen")
        label_map = {"positive": "Positif", "negative": "Negatif", "neutral": "Netral"}
        color_map = {"Positif": "#2f9e44", "Negatif": "#e34948", "Netral": "#f0ad4e"}

        counts = df["sentimen"].map(label_map).value_counts()

        fig = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Contoh Ulasan Terbaru Dianalisis")
        label_display = {"positive": "🟢 Positif", "negative": "🔴 Negatif", "neutral": "🟡 Netral"}
        sample_df = get_sample_reviews(df, per_class=2)
        for _, row in sample_df.iterrows():
            teks = str(row["content"])[:100] + ("..." if len(str(row["content"])) > 100 else "")
            st.write(f"{label_display.get(row['sentimen'], row['sentimen'])} — {teks}")

    st.divider()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("📊 Lihat Proses Modeling", use_container_width=True):
            st.switch_page("pages/modeling.py")
    with col_b:
        if st.button("📄 Lihat Hasil Klasifikasi", use_container_width=True):
            st.switch_page("pages/hasil_klasifikasi.py")
    with col_c:
        if st.button("📤 Upload Data Baru", use_container_width=True):
            st.switch_page("pages/hal_preprocessing.py")

else:
    # belum ada data
    st.info(
        "👋 Belum ada data yang dianalisis. Mulai dengan mengupload dataset ulasan "
        "di halaman **Upload Data**.",
        icon="ℹ️"
    )
    if st.button("📤 Mulai Upload Data", type="primary"):
        st.switch_page("pages/hal_preprocessing.py")