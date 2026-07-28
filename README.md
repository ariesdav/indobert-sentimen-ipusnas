CARA MENJALANKAN APLIKASI
------------------------------------------------------------
LANGKAH-LANGKAH
------------------------------------------------------------
1. Buka PowerShell / Terminal, lalu masuk ke folder project ini.
   Contoh:
   cd E:\skripsi\code\TA_streamlit
2. (Jika BELUM pernah membuat virtual environment / venv)
   Buat virtual environment baru:
   python -m venv venv
3. Aktifkan virtual environment:
   venv\Scripts\activate
   Jika berhasil, di depan baris terminal akan muncul tulisan (venv)
4. Install semua library yang dibutuhkan (hanya perlu sekali, atau
   setiap kali requirements.txt berubah):
   pip install -r requirements.txt
   Catatan: proses ini bisa memakan waktu cukup lama karena ada
   library besar (torch, transformers) untuk model IndoBERT.
5. Jalankan aplikasi:
   streamlit run app.py
6. Tunggu sebentar, nanti otomatis akan terbuka tab baru di browser
   dengan alamat seperti: http://localhost:8501
   Kalau tidak otomatis terbuka, copy-paste alamat tersebut ke browser.
7. Untuk MENGHENTIKAN aplikasi:
   Kembali ke jendela terminal, tekan CTRL + C
   Untuk menjalankan lagi di lain waktu, cukup ulangi dari langkah 1,
   3, dan 5 (tidak perlu ulang membuat venv & install requirements
   lagi, kecuali venv dihapus atau requirements.txt berubah).
