#cetak_laporan.py
import streamlit as st
from datetime import datetime
from utils.report_generator import generate_pdf, generate_excel
from zoneinfo import ZoneInfo

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

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(
        """
        <div style="text-align:center; padding: 40px; border: 2px dashed #ccc; border-radius: 10px;">
            <p style="font-size: 48px; margin: 0;">📄</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("Siap Mengunduh Laporan")
    st.write("Laporan berisi:")
    st.markdown(
        """
        - Ringkasan Statistik (Total Sentimen, Total Ulasan)
        - Tabel Data Ulasan
        - Detail Confidence per Ulasan (khusus Excel)
        """
    )

    format_pilihan = st.radio(
        "Pilih Format",
        options=["PDF Document (.pdf)", "Excel Spreadsheet (.xlsx)"],
        index=0,
    )

    # batas baris cuma buat PDF, excel full data
    jumlah_baris = 25
    if "PDF" in format_pilihan:
        jumlah_baris = st.selectbox(
            "Jumlah ulasan ditampilkan di tabel PDF",
            options=[25, 50, 100],
            index=0,
            help="Excel selalu berisi seluruh data, batas ini hanya berlaku untuk PDF agar tetap ringkas dan mudah dibaca."
        )

    if st.button("⬇️ Buat & Unduh Laporan", type="primary"):
        with st.spinner("Menyusun laporan..."):
            if "PDF" in format_pilihan:
                buffer = generate_pdf(df, max_rows=jumlah_baris)
                st.download_button(
                    label="📥 Unduh Laporan PDF",
                    data=buffer,
                    file_name=f"laporan_sentimen_{datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
            else:
                buffer = generate_excel(df)
                st.download_button(
                    label="📥 Unduh Laporan Excel",
                    data=buffer,
                    file_name=f"laporan_sentimen_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🏠 Kembali ke Beranda", use_container_width=True):
        st.switch_page("pages/beranda.py")
with col_b:
    if st.button("📤 Analisis Data Baru", use_container_width=True):
        st.switch_page("pages/hal_preprocessing.py")