"""Daftar cek OCR dua panduan BBK: semua baris chunk yang memuat angka atau kata syarat,
untuk dicocokkan manual dengan PDF aslinya.

    python src/cek_ocr.py            # menulis data/processed/cek_ocr_bbk.csv

Kolom `sesuai_pdf (y/n)` dan `koreksi` diisi tangan. Koreksi diterapkan lewat `fixes`
di configs/ingest_docs.yaml (bukan dengan mengedit chunk), lalu notebook 01-02 dijalankan ulang.
"""
import csv
import json
import re
from pathlib import Path

KRITIS = re.compile(r"\d|wajib|minimal|maksimal|paling lambat|TOEFL|IELTS|Duolingo|paspor|passport", re.I)
ROOT = Path(__file__).resolve().parent.parent


def main(chunks_path=ROOT / "data/processed/chunks.jsonl", out_path=ROOT / "data/processed/cek_ocr_bbk.csv"):
    chunks = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM supaya Excel membaca UTF-8
        w = csv.writer(f)
        w.writerow(["dokumen", "halaman", "baris", "sesuai_pdf (y/n)", "koreksi"])
        for c in chunks:
            if "panduan-bbk-jenis" not in c["doc_id"]:
                continue
            for line in c["text"].split("\n"):
                s = line.strip()
                if len(s) > 8 and KRITIS.search(s) and not s.startswith("(lanjutan)"):
                    w.writerow([c["doc_id"], f'{c["page_start"]}-{c["page_end"]}', s, "", ""])
                    n += 1
    print(f"{n} baris ditulis ke {out_path}")


if __name__ == "__main__":
    main()
