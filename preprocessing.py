# 2. preprocessing.py
#import library
import re
import os
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#Download resource NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


#Resource Loading
#Path ke kamus slang
SLANG_DICT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "slang_dict.csv"
)

_slang_df = pd.read_csv(SLANG_DICT_PATH)
_slang_formal_map = dict(zip(_slang_df["slang"], _slang_df["formal"]))

#Stopwords Indonesia + Inggris
_stopwords_indo = set(stopwords.words("indonesian"))
_stopwords_eng = set(stopwords.words("english"))
_combined_stopwords = _stopwords_indo.union(_stopwords_eng)
_combined_stopwords.update({"btw", "wkwk", "hehe", "haha", "loh", "yah", "kok", "gue", "lu"})
_negation_words = {"tidak", "tak", "bukan", "ga", "gak", "nggak", "enggak", "belum"}
_combined_stopwords = _combined_stopwords - _negation_words

#Stemmer Sastrawi
_stemmer_factory = StemmerFactory()
_stemmer = _stemmer_factory.create_stemmer()

#exception, kata yang kalau distem penuh bisa mengubah makna
_stem_exceptions = {
    "diperbaiki": "masalah",
    "memperbaiki": "masalah",
    "diperbaikinya": "masalah",
    "perbaikan": "masalah",
    "dibenerin": "masalah",
    "dibetulin": "masalah",
}

# Cache hasil stemming per kata
_stem_cache = {}

#Fungsi-fungsi tahap preprocessing
def caseFoldingText(text):
    """Ubah teks jadi lowercase."""
    return str(text).lower()


def cleaningText(text):
    """Hapus mention, hashtag, URL, angka, tanda baca, emoji, dan rapikan spasi."""
    text = str(text)
    text = re.sub(r"@\w+", "", text)          # hapus mention
    text = re.sub(r"#\w+", "", text)          # hapus hashtag
    text = re.sub(r"http\S+", "", text)       # hapus URL
    text = re.sub(r"\d+", "", text)           # hapus angka
    text = re.sub(r"[^\w\s]", "", text)       # hapus tanda baca
    text = re.sub(r"[^\x00-\x7f]", "", text)  # hapus emoji
    text = re.sub(r"\s+", " ", text).strip()  # rapikan spasi
    return text


def replaceSlang(text):
    """Ganti kata slang jadi kata baku berdasarkan slang_dict.csv."""
    words = text.split()
    replaced_words = [_slang_formal_map.get(word, word) for word in words]
    return " ".join(replaced_words)


def tokenizingText(text):
    """Pecah teks jadi list token."""
    return word_tokenize(text)


def filteringText(tokens):
    """Buang stopword, tapi pertahankan kata negasi."""
    return [word for word in tokens if word not in _combined_stopwords]


def stemmingText(tokens):
    """
    Ubah satu list token ke bentuk dasar (stemming) pakai Sastrawi, dengan
    cache & pengecualian kata tertentu.
    """
    result = []
    for w in tokens:
        wl = w.lower()
        if wl in _stem_exceptions:
            result.append(_stem_exceptions[wl])
            continue
        if wl in _stem_cache:
            result.append(_stem_cache[wl])
        else:
            stemmed = _stemmer.stem(w)
            _stem_cache[wl] = stemmed
            result.append(stemmed)
    return result


def stem_dataframe_column(filtered_series: pd.Series) -> pd.Series:
    """
    Stem satu kolom penuh (Series berisi list token) secara efisien.
    """
    all_words = set()
    for tokens in filtered_series:
        all_words.update(w.lower() for w in tokens)

    words_to_stem = [w for w in all_words if w not in _stem_cache and w not in _stem_exceptions]

    for w in words_to_stem:
        _stem_cache[w] = _stemmer.stem(w)

    def _map_tokens(tokens):
        return [
            _stem_exceptions.get(w.lower(), _stem_cache.get(w.lower(), w))
            for w in tokens
        ]

    return filtered_series.apply(_map_tokens)


def toSentence(tokens):
    """Gabungkan list token jadi satu string kalimat lagi."""
    return " ".join(tokens)

#Pipeline utama
def preprocess_pipeline(df: pd.DataFrame, text_col: str = "content") -> pd.DataFrame:
    """
    Jalankan seluruh tahap preprocessing pada kolom teks di DataFrame,
    sampai tahap stemming sesuai notebook labeling.
    """
    if text_col not in df.columns:
        raise ValueError(f"Kolom '{text_col}' tidak ditemukan dalam data.")

    df = df.copy()

    # Handle missing/kosong dahulu, agar tidak error di tahap berikutnya
    df[text_col] = df[text_col].replace("", pd.NA).replace(" ", pd.NA)
    df[text_col] = df[text_col].fillna("unknown")

    df["case_folding"] = df[text_col].apply(caseFoldingText)
    df["cleaning"] = df["case_folding"].apply(cleaningText)
    df["slang"] = df["cleaning"].apply(replaceSlang)
    df["tokenizing"] = df["slang"].apply(tokenizingText)
    df["filtered"] = df["tokenizing"].apply(filteringText)
    df["stemmed"] = stem_dataframe_column(df["filtered"])
    df["final_text"] = df["stemmed"].apply(toSentence)

    return df


def preprocess_single_text(text: str) -> str:
    """
    Jalankan pipeline preprocessing penuh untuk SATU teks saja,
    dipakai di halaman 'Uji Klasifikasi Manual', sebelum masuk ke model.
    """
    if not text or not str(text).strip():
        return ""

    step = caseFoldingText(text)
    step = cleaningText(step)
    step = replaceSlang(step)
    tokens = tokenizingText(step)
    tokens = filteringText(tokens)
    tokens = stemmingText(tokens)
    return toSentence(tokens)