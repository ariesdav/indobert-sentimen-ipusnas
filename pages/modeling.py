#modelin.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

# paksa scroll ke atas tiap kali halaman ini dibuka
components.html(
    """
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
    </script>
    """,
    height=0
)

st.title("🧠 Modeling — Hasil Training IndoBERT")

st.info(
    "Halaman ini menampilkan hasil evaluasi model yang sudah dilatih (fine-tuning IndoBERT), "
    "bukan analisis dari data yang kamu upload di halaman lain dam angka di sini bersifat tetap, "
    "hasil dari proses training model.",
    icon="ℹ️"
)

# 1. Info Dataset & Setup Training
st.subheader("1. Dataset & Setup Training")

col1, col2, col3 = st.columns(3)
col1.metric("Total Data (sebelum split)", "3.513")
col2.metric("Data Train (setelah augmentasi)", "4.084")
col3.metric("Data Test", "703")

dist_awal = pd.DataFrame({
    "Kelas": ["Negatif", "Netral", "Positif"],
    "Jumlah Data Awal": [2043, 1065, 405],
})
st.dataframe(dist_awal, use_container_width=True, hide_index=True)

st.markdown(
    """
    - Label sentimen menggunakan hasil **labeling berbasis model IndoBERT**,
      karena secara kualitatif lebih akurat menangkap konteks kalimat (negasi, kata slang/teknis,
      dan ambiguitas bahasa sehari-hari).
    - Data dibagi **80% train / 20% test** secara *stratified* agar proporsi tiap kelas tetap terjaga di kedua set.
    - Karena kelas **positif** dan **netral** jauh lebih sedikit dibanding **negatif**, dilakukan
      **augmentasi data** (synonym replacement berbasis tesaurus) secara **proporsional** pada train set,
      target augmentasi dihitung otomatis (~75% dari jumlah kelas mayoritas), bukan angka tetap, agar data
      sintetis tidak mendominasi dan menimbulkan overfitting terhadap pola artifisial.
    - Model dilatih menggunakan **class weight (balanced)** pada loss function, ditambah **label smoothing**
      dan **freezing 6 layer awal IndoBERT** untuk menekan overfitting mengingat ukuran dataset yang terbatas
      relatif terhadap ukuran model.
    - Base model: `indobert-base-p2` (IndoBERT), fine-tuned dengan *early stopping* (patience 2, dipilih
      berdasarkan F1 terbaik). Training/validation loss masih menunjukkan sedikit gap di epoch-epoch akhir,
      namun jauh lebih terkendali dibanding percobaan awal yang mengindikasikan overfitting sudah cukup teratasi.
    """
)

st.divider()

# hasil evaluasi model (angka statis dari proses training)
target_names = ["negative", "neutral", "positive"]
label_id_map = {"negative": "Negatif", "neutral": "Netral", "positive": "Positif"}
target_names_id = [label_id_map[t] for t in target_names]

cm = np.array([
    [353, 33, 23],
    [61, 134, 18],
    [11, 5, 65],
])

metrics_df = pd.DataFrame({
    "precision": [0.83, 0.78, 0.61],
    "recall":    [0.86, 0.63, 0.80],
    "f1-score":  [0.85, 0.70, 0.70],
}, index=target_names_id)

# 2. Confusion Matrix
st.subheader("2. Confusion Matrix")

col_a, col_b = st.columns([1, 1])

with col_a:
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names_id, yticklabels=target_names_id,
        ax=ax_cm, cbar=False
    )
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)

with col_b:
    st.markdown(
        """
        **Interpretasi:**

        Model paling akurat memprediksi kelas **negatif** (353 dari 409 data, 86%),
        diikuti kelas **positif** (65 dari 81 data, 80%).

        Kelas **netral** memiliki tingkat kesalahan paling tinggi (134 dari 213 data
        benar, 63%) yang sebagian besar kesalahannya tertukar dengan kelas **negatif**
        (61 data), kemungkinan karena batas makna antara ulasan yang bernada netral
        dan negatif ringan/datar cenderung tipis secara bahasa, bukan semata soal
        jumlah data.
        """
    )

st.divider()

# 3. Precision / Recall / F1 per Kelas
st.subheader("3. Precision, Recall, dan F1-score per Kelas")

col_c, col_d = st.columns([1, 1])

with col_c:
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(target_names_id))
    width = 0.25
    ax_bar.bar(x - width, metrics_df["precision"], width, label="Precision", color="#2a78d6")
    ax_bar.bar(x, metrics_df["recall"], width, label="Recall", color="#e34948")
    ax_bar.bar(x + width, metrics_df["f1-score"], width, label="F1-score", color="#2f9e44")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(target_names_id)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Skor")
    ax_bar.legend()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

with col_d:
    st.markdown(
        """
        **Interpretasi:**

        Kelas **negatif** memperoleh skor precision dan recall paling seimbang
        dan tertinggi secara keseluruhan (F1 = 0.85), didukung jumlah data yang
        paling banyak.

        Kelas **positif**, meski jumlah datanya paling sedikit (81 data di test
        set), berhasil mencapai recall tinggi (0.80) berkat augmentasi data dan
        class weighting, walau precision-nya masih lebih rendah (0.61), juga
        menunjukkan model kadang terlalu "berani" menebak positif.

        Kelas **netral** memiliki performa paling rendah, khususnya **recall**
        (0.63), sejalan dengan confusion matrix di mana banyak data netral
        salah diprediksi sebagai kelas negatif.
        """
    )

    st.metric("Akurasi Keseluruhan Model", "79%")
    st.caption("Macro avg F1: 0.75 · Weighted avg F1: 0.78")

st.dataframe(
    metrics_df.style.format("{:.2f}"),
    use_container_width=True
)