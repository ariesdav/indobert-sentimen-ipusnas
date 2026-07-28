# 1. app.py
import streamlit as st
from utils.branding import LOGO_SVG

# setting halaman web (judul, icon, layout)
st.set_page_config(
    page_title="Analisis Sentimen Ulasan",
    page_icon="assets/logo.png",
    layout="wide",
)

# buat sidebar buat logo & judul aplikasi
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; padding: 10px 0 15px 0; margin-top: -15px;">
            <div style="width:32px; height:32px; flex-shrink:0;">{LOGO_SVG}</div>
            <div>
                <p style="font-size: 13px; color: #ddd; margin: 0; font-weight: 600;">Analisis Sentimen iPusnas</p>
                <p style="font-size: 11px; color: #888; margin: 0;">Model: IndoBERT</p>
            </div>
        </div>
        <hr style="margin: 0 0 10px 0; border-color: #333;">
        """,
        unsafe_allow_html=True,
    )

# daftar halaman-halaman yang ada di aplikasi
beranda = st.Page("pages/beranda.py", title="Beranda", icon="🏠", default=True)
upload = st.Page("pages/hal_preprocessing.py", title="Preprocessing", icon="🧹")
hasil = st.Page("pages/hasil_klasifikasi.py", title="Hasil Klasifikasi", icon="📄")
visualisasi= st.Page("pages/modeling.py", title="Modeling", icon="📊")
uji_manual = st.Page("pages/uji_manual.py", title="Uji Manual", icon="✍️")
cetak_laporan = st.Page("pages/cetak_laporan.py", title="Cetak Laporan", icon="📥")

# navigasi antar halaman selalu dijalankan
pg = st.navigation([beranda, upload, hasil, visualisasi, uji_manual, cetak_laporan])
pg.run()