# Cara Menjalankan Aplikasi

Panduan menjalankan aplikasi Sistem Analisis Sentimen Ulasan (IndoBERT) secara lokal.

## Langkah-Langkah

### 1. Masuk ke folder project

Buka PowerShell / Terminal, lalu masuk ke folder project ini.

```bash
cd E:\skripsi\code\TA_streamlit
```

### 2. Buat virtual environment (hanya sekali di awal)

Kalau belum pernah membuat virtual environment / venv sebelumnya:

```bash
python -m venv venv
```

### 3. Aktifkan virtual environment

```bash
venv\Scripts\activate
```

Jika berhasil, di depan baris terminal akan muncul tulisan `(venv)`.

### 4. Install semua library yang dibutuhkan

```bash
pip install -r requirements.txt
```

> ⚠️ **Catatan:** proses ini bisa memakan waktu cukup lama karena ada library besar (`torch`, `transformers`) untuk model IndoBERT. Langkah ini hanya perlu dilakukan sekali, atau setiap kali `requirements.txt` berubah.

### 5. Jalankan aplikasi

```bash
streamlit run app.py
```

### 6. Buka di browser

Tunggu sebentar, nanti otomatis akan terbuka tab baru di browser dengan alamat seperti:

```
http://localhost:8501
```

Kalau tidak otomatis terbuka, copy-paste alamat tersebut ke browser.

### 7. Menghentikan aplikasi

Kembali ke jendela terminal, tekan `CTRL + C`.

---

## Menjalankan Lagi di Lain Waktu

Setelah setup awal selesai, kamu **tidak perlu** mengulang membuat venv & install requirements lagi (kecuali venv dihapus atau `requirements.txt` berubah). Cukup ulangi:

1. Masuk ke folder project (`cd ...`)
2. Aktifkan venv (`venv\Scripts\activate`)
3. Jalankan aplikasi (`streamlit run app.py`)
