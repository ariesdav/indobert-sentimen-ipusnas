# Sistem Analisis Sentimen Ulasan iPusnas (IndoBERT)

Aplikasi untuk klasifikasi sentimen ulasan pengguna aplikasi iPusnas menggunakan model IndoBERT yang sudah di-fine-tune. Dibangun dengan Streamlit.

## Cara Menjalankan

1. Masuk ke folder project
```bash
cd E:\skripsi\code\TA_streamlit
```

2. Buat virtual environment (kalau belum pernah)
```bash
python -m venv venv
```

3. Aktifkan virtual environment
```bash
venv\Scripts\activate
```
Kalau berhasil, akan muncul `(venv)` di depan baris terminal.

4. Install dependencies
```bash
pip install -r requirements.txt
```
Proses ini lumayan lama karena ada `torch` dan `transformers`. Cukup dilakukan sekali, atau kalau `requirements.txt` berubah.

5. Jalankan aplikasi
```bash
streamlit run app.py
```

6. Buka browser ke `http://localhost:8501` (biasanya kebuka otomatis)

7. Untuk stop, tekan `CTRL + C` di terminal

## Menjalankan Lagi Setelahnya

Tidak perlu ulang buat venv atau install requirements lagi. Cukup:
```bash
cd E:\skripsi\code\TA_streamlit
venv\Scripts\activate
streamlit run app.py
```

## Struktur Project

- `app.py` — entry point aplikasi Streamlit
- `inference.py` — load model dan fungsi prediksi
- `preprocessing.py` — pipeline preprocessing teks
- `pages/` — halaman-halaman aplikasi (upload, hasil klasifikasi, modeling, dll)
- `label_encoder.pkl` — encoder label sentimen
