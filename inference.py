# 3. inference.py
#import library
import os
import pickle

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Konfigurasi path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "ariesdav/indobert-sentimen-ipusnas"
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

#cek gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128


def load_model():
    """
    Load tokenizer, model, dan label encoder dari folder model_final/.
    """
    # load tokenizer & model dari hasil training
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    # load label encoder buat ubah angka jadi label
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "label_encoder": label_encoder,
    }


def _decode_label(label_encoder, class_id: int):
    """
    Ubah class_id (0/1/2) jadi label teks ('positive'/'neutral'/'negative').
    """
    if hasattr(label_encoder, "inverse_transform"):
        return label_encoder.inverse_transform([class_id])[0]
    elif isinstance(label_encoder, dict):
        return label_encoder.get(class_id, str(class_id))
    else:
        # fallback terakhir, biar tidak silent-fail
        return str(class_id)


def _all_labels(label_encoder, num_labels: int):
    """
    Ambil daftar nama label untuk semua class_id.
    """
    return [_decode_label(label_encoder, i) for i in range(num_labels)]


@torch.no_grad()
def predict_single(text: str, resources: dict) -> dict:
    """
    Prediksi sentimen untuk satu teks (sudah dipreprocess).
    """
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    label_encoder = resources["label_encoder"]

    # tokenisasi teks input jadi format yang bisa dibaca model
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    # ambil hasil prediksi & ubah jadi probabilitas
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    class_id = int(torch.argmax(probs).item())
    confidence = float(probs[class_id].item())
    label = _decode_label(label_encoder, class_id)

    label_names = _all_labels(label_encoder, probs.shape[0])
    probs_dict = {label_names[i]: float(p) for i, p in enumerate(probs.tolist())}

    return {"label": label, "confidence": confidence, "probs": probs_dict}


@torch.no_grad()
def predict_batch(texts: list, resources: dict, batch_size: int = 16) -> list:
    """
    Prediksi sentimen untuk banyak teks sekaligus (dipakai di halaman
    Hasil Klasifikasi, setelah CSV di-upload dan dipreprocess).
    """
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    label_encoder = resources["label_encoder"]

    results = []

    # proses teks per batch supaya tidak berat kalau datanya banyak
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        #Handle teks kosong/NaN agar tidak error saat tokenizing.
        safe_texts = []
        for t in batch_texts:
            s = "" if pd.isna(t) else str(t).strip()
            safe_texts.append(s if s else "unknown")

        inputs = tokenizer(
            safe_texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        ).to(DEVICE)

        logits = model(**inputs).logits
        probs_batch = torch.softmax(logits, dim=-1)

        label_names = _all_labels(label_encoder, probs_batch.shape[1])

        # simpan hasil prediksi tiap baris ke list results
        for row_probs in probs_batch:
            class_id = int(torch.argmax(row_probs).item())
            confidence = float(row_probs[class_id].item())
            label = _decode_label(label_encoder, class_id)
            probs_dict = {label_names[i]: float(p) for i, p in enumerate(row_probs.tolist())}

            results.append({
                "label": label,
                "confidence": confidence,
                "probs": probs_dict,
            })

    return results