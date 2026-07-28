#hal_preprocessing.py
import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import (
    caseFoldingText, cleaningText, replaceSlang,
    tokenizingText, filteringText, stem_dataframe_column, toSentence
)

st.title("🧹 Preprocessing Data")

st.info(
    "Silakan upload file CSV mentah hasil scraping dari Google Play Store, atau upload "
    "data yang sudah diproses sebelumnya (misal dari notebook). Sistem akan melakukan "
    "**Preprocessing Otomatis** melalui seluruh tahapan hingga data siap diklasifikasi.",
    icon="ℹ️"
)

# Urutan tahap preprocessing yang mau ditampilkan
STAGE_COLUMNS = [
    ("content", "1️⃣ Data Mentah"),
    ("case_folding", "2️⃣ Case Folding"),
    ("cleaning", "3️⃣ Cleaning (Regex)"),
    ("slang", "4️⃣ Slang Replacement"),
    ("tokenizing", "5️⃣ Tokenizing"),
    ("filtered", "6️⃣ Filtering (Stopword)"),
    ("stemmed", "7️⃣ Stemming"),
    ("final_text", "8️⃣ Final Text"),
]


def show_wordcloud_section(df: pd.DataFrame):
    """Tampilin wordcloud sebelum vs sesudah preprocessing."""
    st.subheader("☁️ Wordcloud Sebelum vs Sesudah Preprocessing")

    has_raw = "content" in df.columns
    processed_text = " ".join(df["final_text"].dropna().astype(str))

    if not processed_text.strip():
        st.warning("Data teks kosong, wordcloud tidak bisa dibuat.")
        return

    if has_raw:
        raw_text = " ".join(df["content"].astype(str))
        if not raw_text.strip():
            has_raw = False

    with st.spinner("Membuat wordcloud..."):
        after_cloud = WordCloud(width=900, height=500, background_color="white").generate(processed_text)
        before_cloud = None
        if has_raw:
            before_cloud = WordCloud(width=900, height=500, background_color="white").generate(raw_text)

    # kalau data ga punya kolom content, cuma tampilin wordcloud sesudah
    if not has_raw:
        st.caption(
            "ℹ️ Kolom 'content' (data mentah) tidak tersedia pada data ini, "
            "jadi hanya wordcloud sesudah preprocessing yang ditampilkan."
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(after_cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Sesudah Preprocessing", fontsize=13)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.imshow(before_cloud, interpolation="bilinear")
        ax1.axis("off")
        ax1.set_title("Sebelum Preprocessing", fontsize=13)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.imshow(after_cloud, interpolation="bilinear")
        ax2.axis("off")
        ax2.set_title("Sesudah Preprocessing", fontsize=13)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


def run_preprocessing(raw_df: pd.DataFrame):
    """Jalankan seluruh tahap preprocessing dengan timer per tahap, simpan ke session_state."""
    df_work = raw_df.copy()
    df_work["content"] = df_work["content"].replace("", pd.NA).replace(" ", pd.NA).fillna("unknown")

    timings = {}
    progress = st.progress(0, text="Case folding...")

    # tiap tahap dicatat waktunya buat laporan nanti
    t0 = time.time()
    df_work["case_folding"] = df_work["content"].apply(caseFoldingText)
    timings["Case Folding"] = time.time() - t0
    progress.progress(15, text="Cleaning...")

    t0 = time.time()
    df_work["cleaning"] = df_work["case_folding"].apply(cleaningText)
    timings["Cleaning"] = time.time() - t0
    progress.progress(30, text="Slang replacement...")

    t0 = time.time()
    df_work["slang"] = df_work["cleaning"].apply(replaceSlang)
    timings["Slang Replacement"] = time.time() - t0
    progress.progress(45, text="Tokenizing...")

    t0 = time.time()
    df_work["tokenizing"] = df_work["slang"].apply(tokenizingText)
    timings["Tokenizing"] = time.time() - t0
    progress.progress(60, text="Filtering stopword...")

    t0 = time.time()
    df_work["filtered"] = df_work["tokenizing"].apply(filteringText)
    timings["Filtering"] = time.time() - t0
    progress.progress(75, text="Stemming...")

    t0 = time.time()
    df_work["stemmed"] = stem_dataframe_column(df_work["filtered"])
    timings["Stemming"] = time.time() - t0
    progress.progress(95, text="Menyusun final text...")

    t0 = time.time()
    df_work["final_text"] = df_work["stemmed"].apply(toSentence)
    timings["Final Text"] = time.time() - t0
    progress.progress(100, text="Selesai!")
    progress.empty()

    st.session_state["raw_df"] = raw_df
    st.session_state["processed_df"] = df_work

    total_waktu = sum(timings.values())
    with st.expander(f"⏱️ Detail waktu proses per tahap (total: {total_waktu:.2f} detik)"):
        for tahap, detik in timings.items():
            st.write(f"{tahap}: **{detik:.2f} detik**")

    st.success("✅ Preprocessing selesai! Data siap diklasifikasi.")


def load_processed_upload(processed_df: pd.DataFrame):
    """
    Animasi loading dummy untuk file yang udah hasil preprocessing (final_text
    sudah ada).
    """
    dummy_steps = [
        (15, "Memvalidasi kolom..."),
        (30, "Memuat data mentah..."),
        (45, "Memuat hasil cleaning..."),
        (60, "Memuat hasil slang replacement..."),
        (75, "Memuat hasil tokenizing & filtering..."),
        (90, "Memuat hasil stemming..."),
        (100, "Menyusun final text..."),
    ]

    progress = st.progress(0, text="Memuat data hasil preprocessing...")
    for pct, label in dummy_steps:
        time.sleep(0.15)
        progress.progress(pct, text=label)
    progress.empty()

    st.session_state["raw_df"] = processed_df
    st.session_state["processed_df"] = processed_df

    st.success(
        f"✅ Data hasil preprocessing berhasil dimuat — {len(processed_df)} baris siap diklasifikasi."
    )

# Pilihan sumber data
sumber_data = st.radio(
    "Sumber Data",
    options=[
        "📤 Upload File Sendiri",
        "📊 Gunakan Data Contoh (Sudah Dianalisis)",   
    ],
    horizontal=True,
)

raw_df = None

#Path file contoh yang sudah lengkap
SAMPLE_ANALYZED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "resources", "finalipusnasdata.csv"
)

if sumber_data == "📤 Upload File Sendiri":
    # reset session state kalau baru pindah dari mode sample ke upload
    if st.session_state.get("data_source") != "upload":
        st.session_state.pop("processed_df", None)
        st.session_state.pop("raw_df", None)
        st.session_state.pop("classified_df", None)
        st.session_state.pop("sample_data_loaded", None)
        st.session_state["data_source"] = "upload"

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["csv"],
        help="Limit 200MB per file • CSV"
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
            st.stop()

        if "content" not in raw_df.columns:
            st.error(
                "Kolom 'content' tidak ditemukan di file yang diupload. "
                "Pastikan file CSV memiliki kolom bernama 'content' berisi teks ulasan."
            )
            st.stop()

        st.success(f"File berhasil diupload — {len(raw_df)} baris data ditemukan.")

elif sumber_data == "📊 Gunakan Data Contoh (Sudah Dianalisis)":
    st.session_state["data_source"] = "sample"
    st.caption(
        "Menggunakan data contoh ulasan ipusnas yang sudah melewati seluruh tahap "
        "preprocessing dan sudah diklasifikasikan modelnya sebelumnya. Tinggal lanjut "
        "ke halaman berikutnya, tanpa perlu upload atau menunggu proses apapun."
    )

    if not os.path.exists(SAMPLE_ANALYZED_PATH):
        st.error(
            f"File data contoh tidak ditemukan di `{SAMPLE_ANALYZED_PATH}`. "
            "Pastikan file `finalipusnasdata.csv` sudah ditaruh di folder `resources/`."
        )
        st.stop()

    sample_df = pd.read_csv(SAMPLE_ANALYZED_PATH)

    #Hanya load sekali supaya tidak keulang animasi tiap rerun
    if st.session_state.get("sample_data_loaded") != True:
        st.session_state["raw_df"] = sample_df
        st.session_state["processed_df"] = sample_df
        st.session_state["sample_data_loaded"] = True
        st.session_state.pop("classified_df", None)

    st.success(f"✅ Data contoh berhasil dimuat — {len(sample_df)} baris siap ditampilkan.")

if raw_df is not None:
    with st.expander("👀 Lihat data mentah (5 baris pertama)"):
        st.dataframe(raw_df.head(), use_container_width=True)

    if st.button("🚀 Jalankan Preprocessing", type="primary"):
        run_preprocessing(raw_df)

if "processed_df" in st.session_state:
    df = st.session_state["processed_df"]
    available_stages = [(col, label) for col, label in STAGE_COLUMNS if col in df.columns]

    st.subheader("Preview Tiap Tahap Preprocessing")

    tabs = st.tabs([label for _, label in available_stages])
    for tab, (col, label) in zip(tabs, available_stages):
        with tab:
            st.dataframe(df[[col]].head(10), use_container_width=True)

    with st.expander("📋 Lihat semua tahap sekaligus (tabel gabungan)"):
        st.dataframe(df[[col for col, _ in available_stages]].head(10), use_container_width=True)

    st.divider()
    show_wordcloud_section(df)

    if st.button("➡️ Lanjut ke halaman Hasil Klasifikasi untuk melihat prediksi sentimen.", type="primary"):
        st.switch_page("pages/hasil_klasifikasi.py")