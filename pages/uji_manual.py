#uji_manual.py
import streamlit as st
from preprocessing import preprocess_single_text
from inference import predict_single

st.set_page_config(page_title="Uji Manual", layout="wide")

st.title("✍️ Uji Klasifikasi Manual")

st.write(
    "Masukkan satu kalimat ulasan untuk diuji langsung tanpa perlu upload file CSV."
)

# guard clause: pastikan model sudah dimuat
if "resources" not in st.session_state:
    st.error("Model belum dimuat. Silakan kembali ke halaman utama (app) terlebih dahulu.")
    st.stop()

resources = st.session_state["resources"]

# input
input_text = st.text_area(
    "Teks ulasan",
    placeholder="Contoh: aplikasinya bagus banget, sangat membantu buat baca buku",
    height=120,
)

analyze_clicked = st.button("🔍 Analisis Sentimen", type="primary")

st.divider()

if analyze_clicked:
    if not input_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.", icon="⚠️")
        st.stop()

    # preprocessing dulu sebelum masuk model
    with st.spinner("Menganalisis..."):
        clean_text = preprocess_single_text(input_text)
        result = predict_single(clean_text, resources)

    label = result["label"]
    confidence = result["confidence"]
    probs = result["probs"]

    # badge warna sesuai label
    label_style = {
        "positive": {"color": "#2f9e44", "bg": "#d4edda", "text": "Positif"},
        "negative": {"color": "#e34948", "bg": "#f8d7da", "text": "Negatif"},
        "neutral":  {"color": "#856404", "bg": "#fff3cd", "text": "Netral"},
    }
    style = label_style.get(label, {"color": "#333", "bg": "#eee", "text": label})

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {style['bg']};
                        text-align: center; border: 1px solid {style['color']};">
                <p style="margin: 0; font-size: 14px; color: #555;">Hasil Prediksi</p>
                <p style="margin: 0; font-size: 32px; font-weight: bold; color: {style['color']};">
                    {style['text']}
                </p>
                <p style="margin: 0; font-size: 16px; color: #555;">
                    Confidence: {confidence*100:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.write("**Breakdown Probabilitas**")
        label_display = {"positive": "Positif", "negative": "Negatif", "neutral": "Netral"}
        for cls, prob in probs.items():
            display_name = label_display.get(cls, cls)
            st.write(f"{display_name}")
            st.progress(prob, text=f"{prob*100:.1f}%")

    st.divider()

    with st.expander("🔧 Lihat detail preprocessing"):
        st.write("**Teks asli:**")
        st.code(input_text, language=None)
        st.write("**Teks setelah preprocessing (yang masuk ke model):**")
        st.code(clean_text, language=None)