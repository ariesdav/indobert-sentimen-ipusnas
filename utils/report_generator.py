# 4. report_generator.py
# Modul untuk generate laporan (PDF & Excel) dari hasil klasifikasi sentimen.
import os
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

LABEL_MAP = {"positive": "Positif", "negative": "Negatif", "neutral": "Netral"}
SENTIMEN_ORDER = ["Negatif", "Netral", "Positif"]

BULAN_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}

# Path logo untuk kop surat PDF.
LOGO_PATH = "assets/logo.png"

# Urutan tahap preprocessing yang ditampilkan di section "Tahapan Preprocessing"
PREPROCESSING_STAGE_COLUMNS = [
    ("content", "1. Data Mentah"),
    ("case_folding", "2. Case Folding"),
    ("cleaning", "3. Cleaning (Regex)"),
    ("slang", "4. Slang Replacement"),
    ("tokenizing", "5. Tokenizing"),
    ("filtered", "6. Filtering (Stopword)"),
    ("stemmed", "7. Stemming"),
    ("final_text", "8. Final Text"),
]

# Data statis performa model
MODEL_TARGET_NAMES_ID = ["Negatif", "Netral", "Positif"]

MODEL_CONFUSION_MATRIX = np.array([
    [197, 17, 58],
    [15, 90, 16],
    [26, 19, 265],
])

MODEL_METRICS_DF = pd.DataFrame({
    "precision": [0.83, 0.71, 0.78],
    "recall":    [0.72, 0.74, 0.85],
    "f1-score":  [0.77, 0.73, 0.82],
}, index=MODEL_TARGET_NAMES_ID)

MODEL_ACCURACY = "79%"

MODEL_DATASET_INFO = [
    ("Total Data (sebelum split)", "3.513"),
    ("Data Train (setelah augmentasi)", "4.482"),
    ("Data Test", "703"),
]

MODEL_CM_INTERPRETASI = (
    "Model paling akurat memprediksi kelas <b>positif</b> (265 dari 310 data) dan "
    "<b>negatif</b> (197 dari 272 data). Kelas <b>netral</b> memiliki tingkat kesalahan "
    "paling tinggi &mdash; sebagian besar kesalahannya tertukar dengan kelas negatif "
    "(26 data) dibanding kelas positif (19 data), kemungkinan karena jumlah data netral "
    "yang jauh lebih sedikit dibanding kedua kelas lainnya."
)

MODEL_METRICS_INTERPRETASI = (
    "Kelas <b>positif</b> memperoleh skor precision, recall, dan F1-score tertinggi, "
    "diikuti kelas negatif. Kelas <b>netral</b> memiliki performa paling rendah pada "
    "ketiga metrik, khususnya recall (0.74) &mdash; sejalan dengan confusion matrix di "
    "mana banyak data netral salah diprediksi sebagai kelas negatif."
)


def format_tanggal_indonesia(dt: datetime) -> str:
    """Format datetime jadi string tanggal Indonesia, tanpa gantung ke locale server."""
    return f"{dt.day} {BULAN_ID[dt.month]} {dt.year}, {dt.strftime('%H:%M')}"


def _chart_proporsi_sentimen_pie(df: pd.DataFrame) -> BytesIO:
    """Donut chart proporsi sentimen, gaya sama seperti pie chart di halaman beranda.py."""
    color_map = {"Positif": "#2f9e44", "Negatif": "#e34948", "Netral": "#f0ad4e"}
    counts = df["sentimen"].map(LABEL_MAP).value_counts()
    labels = counts.index.tolist()
    colors_list = [color_map[label] for label in labels]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    wedges, _texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors_list,
        autopct="%1.1f%%",
        pctdistance=0.78,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white"),
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
    ax.set_title("Proporsi Sentimen", fontsize=13, fontweight="bold")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _caption_proporsi_sentimen(df: pd.DataFrame) -> str:
    counts = df["sentimen"].map(LABEL_MAP).value_counts().reindex(SENTIMEN_ORDER).fillna(0)
    dominan = counts.idxmax()
    persen_dominan = counts.max() / counts.sum() * 100
    return (
        f"Grafik menunjukkan distribusi sentimen dominan <b>{dominan.lower()}</b> "
        f"({persen_dominan:.1f}% dari total ulasan)."
    )


def _chart_wordcloud_before_after(df: pd.DataFrame, sample_size: int = 1000) -> BytesIO:
    """Wordcloud sebelum vs sesudah preprocessing, side-by-side, untuk section Preprocessing di PDF."""
    if "content" not in df.columns or "final_text" not in df.columns:
        return None

    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
    raw_text = " ".join(sample_df["content"].astype(str))
    processed_text = " ".join(sample_df["final_text"].dropna().astype(str))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    if raw_text.strip():
        wc_before = WordCloud(
            width=700, height=420, background_color="white",
            max_words=80, collocations=False,
        ).generate(raw_text)
        axes[0].imshow(wc_before, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Sebelum Preprocessing", fontsize=10, fontweight="bold")

    if processed_text.strip():
        wc_after = WordCloud(
            width=700, height=420, background_color="white",
            max_words=80, collocations=False,
        ).generate(processed_text)
        axes[1].imshow(wc_after, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Sesudah Preprocessing", fontsize=10, fontweight="bold")

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_preprocessing_section(df: pd.DataFrame, styles, cell_style, caption_style) -> list:
    """
    Bangun section 'Tahapan Preprocessing': wordcloud sebelum vs sesudah
    """
    elements = []

    elements.append(Paragraph("Tahapan Preprocessing", styles["Heading2"]))
    elements.append(Paragraph(
        "Setiap ulasan mentah melalui beberapa tahap pembersihan dan normalisasi teks "
        "sebelum diklasifikasi oleh model.",
        caption_style
    ))
    elements.append(Spacer(1, 8))

    wc_buf = _chart_wordcloud_before_after(df)
    if wc_buf is not None:
        wc_img = RLImage(wc_buf, width=16 * cm, height=6.4 * cm)
        wc_img.hAlign = "CENTER"
        elements.append(wc_img)
        elements.append(Spacer(1, 14))

    available_stages = [(c, l) for c, l in PREPROCESSING_STAGE_COLUMNS if c in df.columns]
    if available_stages and len(df) > 0:
        elements.append(Paragraph("Contoh Tahapan Preprocessing (1 Ulasan)", styles["Heading3"]))
        elements.append(Spacer(1, 4))

        sample_row = df.iloc[0]
        stage_rows = [["Tahap", "Hasil"]]
        for col, label in available_stages:
            val = sample_row[col]
            text = " ".join(val) if isinstance(val, list) else str(val)
            stage_rows.append([label, Paragraph(text, cell_style)])

        stage_table = Table(stage_rows, colWidths=[4 * cm, 11 * cm], repeatRows=1)
        stage_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(stage_table)

    elements.append(Spacer(1, 20))
    return elements


def _chart_confusion_matrix() -> BytesIO:
    """Heatmap confusion matrix, data statis dari hasil evaluasi training (modeling.py)."""
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(
        MODEL_CONFUSION_MATRIX, annot=True, fmt="d", cmap="Blues",
        xticklabels=MODEL_TARGET_NAMES_ID, yticklabels=MODEL_TARGET_NAMES_ID,
        ax=ax, cbar=False
    )
    ax.set_xlabel("Prediksi", fontsize=9)
    ax.set_ylabel("Aktual", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_per_class_metrics() -> BytesIO:
    """Bar chart precision/recall/f1 per kelas, data statis dari hasil evaluasi training."""
    x = np.arange(len(MODEL_TARGET_NAMES_ID))
    width = 0.25

    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.bar(x - width, MODEL_METRICS_DF["precision"], width, label="Precision", color="#2a78d6")
    ax.bar(x, MODEL_METRICS_DF["recall"], width, label="Recall", color="#e34948")
    ax.bar(x + width, MODEL_METRICS_DF["f1-score"], width, label="F1-score", color="#2f9e44")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_TARGET_NAMES_ID, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Skor", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_letterhead(styles) -> list:
    """
    Bangun blok kop surat: logo, judul laporan, judul skripsi/topik, garis pembatas,
    lalu tanggal cetak.
    """
    elements = []

    title_style = ParagraphStyle(
        "KopTitle", parent=styles["Heading1"], fontSize=16, alignment=1,
        spaceAfter=4, leading=20,
    )
    subtitle_style = ParagraphStyle(
        "KopSubtitle", parent=styles["Normal"], fontSize=11, alignment=1,
        textColor=colors.HexColor("#333333"), leading=14,
    )
    date_style = ParagraphStyle(
        "TanggalCetak", parent=styles["Normal"], fontSize=9, alignment=2,
        textColor=colors.HexColor("#777777"),
    )

    if os.path.exists(LOGO_PATH):
        logo_img = RLImage(LOGO_PATH, width=2.1 * cm, height=2.1 * cm)
        logo_img.hAlign = "CENTER"
        elements.append(logo_img)
        elements.append(Spacer(1, 6))

    elements.append(Paragraph("LAPORAN ANALISIS SENTIMEN ULASAN APLIKASI", title_style))
    elements.append(Paragraph(
        "Analisis Sentimen terhadap Ulasan Aplikasi iPusnas di Google Play Store "
        "dengan Metode IndoBERT",
        subtitle_style
    ))
    elements.append(Spacer(1, 10))

    # garis pembatas, gaya kop surat
    divider = Table([[""]], colWidths=[17 * cm])
    divider.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, colors.HexColor("#333333")),
    ]))
    elements.append(divider)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(f"Dibuat pada: {format_tanggal_indonesia(datetime.now())}", date_style))
    elements.append(Spacer(1, 14))

    return elements


def _build_model_performance_section(styles, caption_style) -> list:
    """
    Bangun section 'Performa Model' berisi info dataset/training singkat,
    confusion matrix, dan precision/recall/F1 per kelas (sama seperti halaman modeling.py).
    """
    elements = []

    elements.append(Paragraph("Performa Model", styles["Heading2"]))
    elements.append(Paragraph(
        "Bagian ini menampilkan hasil evaluasi model yang sudah dilatih (fine-tuning IndoBERT) "
        "pada data test, bersifat tetap dan tidak dihitung ulang dari data yang diklasifikasi.",
        caption_style
    ))
    elements.append(Spacer(1, 8))

    # Info dataset & training ringkas
    info_rows = [["Keterangan", "Nilai"]] + [[k, v] for k, v in MODEL_DATASET_INFO]
    info_rows.append(["Akurasi Keseluruhan Model", MODEL_ACCURACY])
    info_table = Table(info_rows, colWidths=[9 * cm, 5 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 16))

    # Confusion matrix + interpretasi
    cm_img = RLImage(_chart_confusion_matrix(), width=7.5 * cm, height=6.5 * cm)
    cm_caption = Paragraph(MODEL_CM_INTERPRETASI, caption_style)
    cm_block = Table([[cm_img, cm_caption]], colWidths=[7.5 * cm, 9 * cm])
    cm_block.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
    ]))
    elements.append(Paragraph("Confusion Matrix", styles["Heading3"]))
    elements.append(Spacer(1, 4))
    elements.append(cm_block)
    elements.append(Spacer(1, 16))

    # Precision/recall/f1 + interpretasi
    metrics_img = RLImage(_chart_per_class_metrics(), width=7.5 * cm, height=6.5 * cm)
    metrics_caption = Paragraph(MODEL_METRICS_INTERPRETASI, caption_style)
    metrics_block = Table([[metrics_img, metrics_caption]], colWidths=[7.5 * cm, 9 * cm])
    metrics_block.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
    ]))
    elements.append(Paragraph("Precision, Recall, dan F1-score per Kelas", styles["Heading3"]))
    elements.append(Spacer(1, 4))
    elements.append(metrics_block)
    elements.append(Spacer(1, 10))

    # Tabel angka metrik per kelas
    metrics_table_rows = [["Kelas", "Precision", "Recall", "F1-score"]]
    for kelas in MODEL_TARGET_NAMES_ID:
        row = MODEL_METRICS_DF.loc[kelas]
        metrics_table_rows.append([
            kelas, f"{row['precision']:.2f}", f"{row['recall']:.2f}", f"{row['f1-score']:.2f}"
        ])
    metrics_table = Table(metrics_table_rows, colWidths=[4 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(metrics_table)

    return elements


def generate_pdf(df: pd.DataFrame, max_rows: int = 25) -> BytesIO:
    """
    Generate laporan PDF berisi: ringkasan statistik, tahapan preprocessing,
    proporsi sentimen, tabel data ulasan, dan performa model.
    """
    total = len(df)
    count_positif = int((df["sentimen"] == "positive").sum())
    count_negatif = int((df["sentimen"] == "negative").sum())
    count_netral = int((df["sentimen"] == "neutral").sum())

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "CellText", parent=styles["Normal"], fontSize=8, leading=10, wordWrap="CJK"
    )
    caption_style = ParagraphStyle(
        "CaptionStyle", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#555555"), leading=11,
    )
    elements = []

    # Kop surat: logo, judul laporan, judul topik, garis pembatas, tanggal cetak
    elements.extend(_build_letterhead(styles))

    # Ringkasan statistik
    elements.append(Paragraph("Ringkasan Statistik", styles["Heading2"]))
    summary_data = [
        ["Kategori", "Jumlah", "Persentase"],
        ["Total Ulasan", str(total), "100%"],
        ["Positif", str(count_positif), f"{count_positif/total*100:.1f}%"],
        ["Negatif", str(count_negatif), f"{count_negatif/total*100:.1f}%"],
        ["Netral", str(count_netral), f"{count_netral/total*100:.1f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[6 * cm, 4 * cm, 4 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Tahapan Preprocessing (wordcloud sebelum/sesudah + contoh progres 1 ulasan)
    elements.extend(_build_preprocessing_section(df, styles, cell_style, caption_style))

    # Proporsi Sentimen (pie chart, sama seperti tampilan di halaman beranda.py)
    elements.append(Paragraph("Proporsi Sentimen", styles["Heading2"]))
    pie_img = RLImage(_chart_proporsi_sentimen_pie(df), width=9 * cm, height=7.6 * cm)
    pie_img.hAlign = "CENTER"
    elements.append(pie_img)
    elements.append(Paragraph(_caption_proporsi_sentimen(df), caption_style))
    elements.append(Spacer(1, 20))

    # Tabel data ulasan (dibatasi max_rows, teks di-wrap pakai Paragraph)
    tampil_n = min(max_rows, total)
    elements.append(Paragraph(
        f"Tabel Data Ulasan (menampilkan {tampil_n} dari {total} ulasan)", styles["Heading2"]
    ))
    detail_rows = [["No", "Ulasan", "Sentimen"]]
    for i, row in df.head(max_rows).iterrows():
        ulasan_text = Paragraph(str(row["content"]), cell_style)
        sentimen_text = LABEL_MAP.get(row["sentimen"], row["sentimen"])
        detail_rows.append([str(i + 1), ulasan_text, sentimen_text])

    detail_table = Table(detail_rows, colWidths=[1.5 * cm, 10 * cm, 2.5 * cm], repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (0, 0), 8),
        ("FONTSIZE", (2, 0), (2, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 24))

    # Performa Model (confusion matrix + precision/recall/F1, statis dari training)
    elements.extend(_build_model_performance_section(styles, caption_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def generate_excel(df: pd.DataFrame) -> BytesIO:
    """
    Generate laporan Excel berisi 2 sheet: Ringkasan dan Detail Data (full, tidak dibatasi).
    Excel tidak menyertakan chart gambar.
    """
    total = len(df)
    count_positif = int((df["sentimen"] == "positive").sum())
    count_negatif = int((df["sentimen"] == "negative").sum())
    count_netral = int((df["sentimen"] == "neutral").sum())

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df = pd.DataFrame({
            "Kategori": ["Total Ulasan", "Positif", "Negatif", "Netral"],
            "Jumlah": [total, count_positif, count_negatif, count_netral],
            "Persentase": [
                "100%",
                f"{count_positif/total*100:.1f}%",
                f"{count_negatif/total*100:.1f}%",
                f"{count_netral/total*100:.1f}%",
            ],
        })
        summary_df.to_excel(writer, sheet_name="Ringkasan", index=False)

        export_df = df[["content", "sentimen", "confidence"]].copy()
        export_df["sentimen"] = export_df["sentimen"].map(LABEL_MAP)
        export_df.columns = ["Ulasan", "Sentimen", "Confidence"]
        export_df.to_excel(writer, sheet_name="Detail Data", index=False)

    buffer.seek(0)
    return buffer