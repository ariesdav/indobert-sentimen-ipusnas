#hasil_klasifikasi.py
import streamlit as st
import pandas as pd
import time
import plotly.express as px
from inference import predict_batch

st.title("📄 Hasil Klasifikasi Sentimen")

if "processed_df" not in st.session_state:
    st.warning(
        "Belum ada data yang diupload. Silakan ke halaman **Preprocessing** terlebih dahulu.",
        icon="⚠️"
    )
    st.stop()

if "resources" not in st.session_state:
    st.error("Model belum dimuat. Silakan kembali ke halaman **Beranda** terlebih dahulu.")
    st.stop()

processed_df = st.session_state["processed_df"]
resources = st.session_state["resources"]

# Deteksi kolom label sentimen yang mungkin sudah ada sebelumnya
EXISTING_LABEL_CANDIDATES = ["sentimentLabel", "sentimen", "sentiment", "label"]
VALID_LABELS = {"positive", "negative", "neutral"}


def _find_existing_label_col(df: pd.DataFrame):
    """Cari kolom yang isinya sudah berupa label sentimen valid."""
    for col in EXISTING_LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        normalized = df[col].astype(str).str.strip().str.lower()
        if normalized.isin(VALID_LABELS).mean() >= 0.9:
            return col, normalized
    return None, None


# Flag one-shot: true kalau user baru aja klik "Klasifikasi Ulang"
force_reclassify = st.session_state.pop("force_reclassify", False)

if "classified_df" not in st.session_state:
    existing_col, existing_normalized = (
        (None, None) if force_reclassify else _find_existing_label_col(processed_df)
    )

    if existing_col is not None:
        # Data upload-an udah punya label sentimen kemudia skip model, langsung pakai
        st.info(
            f"⚡ Data ini sudah punya kolom label sentimen (**{existing_col}**), "
            "klasifikasi model dilewati supaya lebih cepat. Klik **'Klasifikasi Ulang'** "
            "kalau tetap mau dijalankan lewat model.",
        )
        result_df = processed_df.copy()
        result_df["sentimen"] = existing_normalized.values

        # Kalau file sumbernya sudah punya kolom confidence asli dari model, pakai itu. 
        if "confidence" in processed_df.columns:
            result_df["confidence"] = pd.to_numeric(processed_df["confidence"], errors="coerce")
        else:
            result_df["confidence"] = pd.NA

        st.session_state["classified_df"] = result_df
        st.session_state["just_classified"] = True

    else:
        texts = processed_df["final_text"].tolist()
        batch_size = 32
        total_batches = (len(texts) + batch_size - 1) // batch_size

        fun_messages = [
            "Membaca ulasan satu per satu... 🔍",
            "Menimbang nada tiap kalimat... ⚖️",
            "Memilah kata positif dan negatif... 🧩",
            "Menyusun hasil klasifikasi... 📊",
        ]

        # proses klasifikasi per batch biar progress bar bisa update
        with st.status("🧠 Model sedang mengklasifikasi ulasan...", expanded=True) as status:
            all_predictions = []
            progress_bar = st.progress(0)

            for i, start in enumerate(range(0, len(texts), batch_size)):
                batch = texts[start:start + batch_size]
                batch_preds = predict_batch(batch, resources, batch_size=batch_size)
                all_predictions.extend(batch_preds)

                pct = (i + 1) / total_batches
                progress_bar.progress(pct)
                msg = fun_messages[i % len(fun_messages)]
                status.write(f"{msg} ({start + len(batch)}/{len(texts)} ulasan)")

            status.update(label="✅ Klasifikasi selesai!", state="complete", expanded=False)

        result_df = processed_df.copy()
        result_df["sentimen"] = [p["label"] for p in all_predictions]
        result_df["confidence"] = [p["confidence"] for p in all_predictions]
        st.session_state["classified_df"] = result_df
        st.session_state["just_classified"] = True

classified_df = st.session_state["classified_df"]

if st.session_state.get("just_classified"):
    st.session_state["just_classified"] = False

col_a, col_b = st.columns([5, 1])
with col_b:
    if st.button("🔄 Klasifikasi Ulang"):
        del st.session_state["classified_df"]
        st.session_state["force_reclassify"] = True
        st.rerun()

total = len(classified_df)
count_positif = (classified_df["sentimen"] == "positive").sum()
count_negatif = (classified_df["sentimen"] == "negative").sum()
count_netral = (classified_df["sentimen"] == "neutral").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Ulasan", f"{total:,}")
c2.metric("Positif", f"{count_positif:,}", delta=f"{count_positif/total*100:.1f}%")
c3.metric("Negatif", f"{count_negatif:,}", delta=f"-{count_negatif/total*100:.1f}%", delta_color="inverse")
c4.metric("Netral", f"{count_netral:,}")

# Chart Proporsi Sentimen
st.subheader("Proporsi Sentimen")

label_map = {"positive": "Positif", "negative": "Negatif", "neutral": "Netral"}
color_map = {"Positif": "#4caf50", "Negatif": "#e34948", "Netral": "#f5c542"}
order = ["Positif", "Negatif", "Netral"]

counts = classified_df["sentimen"].map(label_map).value_counts().reindex(order).fillna(0)

fig = px.bar(
    x=counts.index,
    y=counts.values,
    color=counts.index,
    color_discrete_map=color_map,
    labels={"x": "", "y": "Jumlah Ulasan"},
)
fig.update_layout(showlegend=False, height=320)
st.plotly_chart(fig, use_container_width=True)

dominan = counts.idxmax()
persen_dominan = counts.max() / counts.sum() * 100
st.caption(f"Sentimen dominan: **{dominan}** ({persen_dominan:.1f}% dari total ulasan).")

st.divider()

st.subheader("Detail Data")

filter_sentimen = st.selectbox(
    "Filter Data sesuai sentimen",
    options=["Semua", "positive", "negative", "neutral"],
    index=0
)

display_df = classified_df.copy()
if filter_sentimen != "Semua":
    display_df = display_df[display_df["sentimen"] == filter_sentimen]

display_df["Sentimen"] = display_df["sentimen"].map(label_map)

# Fallback ke final_text kalau kolom 'content' tidak ada
text_col = "content" if "content" in display_df.columns else "final_text"

table_view = display_df[[text_col, "Sentimen", "confidence"]].rename(
    columns={text_col: "Ulasan", "confidence": "Confidence"}
)

def _format_confidence(v):
    if pd.isna(v):
        return "-"
    return f"{v * 100:.1f}%"

table_view["Confidence"] = table_view["Confidence"].apply(_format_confidence)

def highlight_sentimen(val):
    colors = {
        "Positif": "background-color: #d4edda; color: black",
        "Negatif": "background-color: #f8d7da; color: black",
        "Netral": "background-color: #fff3cd; color: black",
        }
    return colors.get(val, "")

st.dataframe(
    table_view.style.map(highlight_sentimen, subset=["Sentimen"]),
    use_container_width=True,
    hide_index=True
)

st.caption(f"Menampilkan {len(display_df)} dari {total} data.")

if st.button("➡️ Lanjut ke halaman Modeling untuk melihat performa model.", type="primary"):
    st.switch_page("pages/modeling.py")