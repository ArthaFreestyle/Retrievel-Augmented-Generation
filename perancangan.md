# Perancangan Program — Sistem & Eksperimen RAG LPMB UNAIR

**Skripsi:** Evaluasi Komparatif Strategi RAG (Closed-Book, Naive RAG, Hybrid + Reranking) serta Validasi Metrik LLM-as-Judge untuk Sistem Tanya-Jawab Berbahasa Indonesia
**Penyusun:** Artha Gandhi Wardhana Aksa — NIM 434231046
**Format proyek:** Jupyter Notebook (`.ipynb`)
**Dokumen ini:** rancangan teknis program yang harus dibuat, sebelum baris kode pertama ditulis.

---

## 1. Prinsip Perancangan

Lima prinsip berikut menurunkan langsung dari klaim ilmiah proposal. Setiap keputusan teknis di dokumen ini dapat ditelusuri ke salah satunya.

| # | Prinsip | Konsekuensi teknis |
|---|---|---|
| P1 | **Ablation harus valid.** Proposal mengklaim selisih kinerja dapat diatribusikan pada komponen retrieval (subbab 3.4). | Ketiga kondisi **wajib** dijalankan oleh satu fungsi runner yang sama, dibedakan hanya oleh objek konfigurasi. Dilarang menyalin-tempel kode pipeline ke tiga notebook terpisah. |
| P2 | **Benchmark adalah kontribusi utama.** | Ground-truth **tidak boleh** diikat ke `chunk_id`. Diikat ke `(doc_id, halaman, kutipan teks)`, lalu dipetakan ke chunk saat evaluasi. Ganti strategi chunking → benchmark tetap hidup. |
| P3 | **Reproducible & murah.** Mode inference-only, laptop konsumen, kuota API terbatas. | Seluruh panggilan Gemini API melewati satu lapis cache berbasis hash prompt. Menjalankan ulang notebook = gratis dan hasilnya identik. |
| P4 | **Notebook adalah laporan, bukan gudang kode.** | Logika inti tinggal di `src/*.py`; notebook memanggil, menampilkan tabel, dan menarasikan. Sel notebook idealnya < 15 baris. |
| P5 | **Metrik otomatis sedang diadili, bukan dipercaya.** | Penilaian manusia dikumpulkan secara **buta** (blind) terhadap kondisi. Skor otomatis tidak boleh terlihat oleh anotator. |

---

## 2. Struktur Direktori

```
skripsi-rag-lpmb/
├── notebooks/                     ← deliverable utama (.ipynb)
│   ├── 00_setup_and_config.ipynb
│   ├── 01_ingest_documents.ipynb
│   ├── 02_preprocess_chunking.ipynb
│   ├── 03_build_indexes.ipynb
│   ├── 04_build_benchmark.ipynb
│   ├── 05_run_experiments.ipynb
│   ├── 06_eval_retrieval.ipynb
│   ├── 07_eval_generation_ragas.ipynb
│   ├── 08_human_annotation_kit.ipynb
│   ├── 09_metric_validation.ipynb
│   ├── 10_analysis_and_figures.ipynb
│   └── 11_prototype_app.ipynb
├── src/                           ← modul yang di-import notebook
│   ├── config.py                  ← dataclass config + loader YAML
│   ├── ingest.py                  ← unduh, ekstraksi PDF, normalisasi
│   ├── chunking.py
│   ├── indexing.py                ← FAISS + BM25
│   ├── retrievers.py              ← Dense, BM25, Hybrid(RRF), Reranker
│   ├── generator.py               ← wrapper Gemini + structured output
│   ├── llm_cache.py               ← cache SQLite untuk panggilan API
│   ├── pipeline.py                ← RUNNER TUNGGAL untuk 3 kondisi (P1)
│   ├── metrics_retrieval.py       ← recall@k, MRR, nDCG
│   ├── metrics_answer.py          ← wrapper RAGAS + abstain
│   ├── evidence_mapping.py        ← span → chunk_id (P2)
│   ├── stats.py                   ← kappa, korelasi, bootstrap
│   └── viz.py                     ← gaya gambar seragam
├── configs/
│   ├── base.yaml
│   ├── cond_a_closed_book.yaml
│   ├── cond_b_naive_rag.yaml
│   └── cond_c_hybrid_rerank.yaml
├── data/
│   ├── raw/pdf/                   ← PDF asli LPMB (jangan diedit)
│   ├── raw/web/                   ← snapshot HTML laman
│   ├── interim/                   ← teks hasil ekstraksi per dokumen
│   ├── processed/
│   │   ├── documents.jsonl
│   │   └── chunks.jsonl
│   └── benchmark/
│       ├── benchmark.jsonl
│       └── annotation_log.csv
├── indexes/{faiss/, bm25/}
├── runs/{cond_a/, cond_b/, cond_c/}predictions.jsonl
├── results/                       ← CSV metrik siap dikutip ke naskah
├── figures/                       ← PNG/SVG untuk Gambar 4.x
├── cache/llm_cache.sqlite
├── requirements.txt
├── .env.example                   ← GEMINI_API_KEY=...  (.env di-gitignore)
└── README.md
```

**Catatan:** `data/raw/` bersifat *read-only* setelah tahap ingesti. Setiap notebook hanya menulis ke direktori keluarannya sendiri, sehingga tidak ada notebook yang merusak masukan notebook lain.

---

## 3. Konvensi Setiap Notebook

Semua notebook memakai kerangka yang sama agar konsisten dan mudah diaudit penguji:

```
[Sel 1 — Markdown]  Judul, tujuan, INPUT (file), OUTPUT (file),
                    perkiraan runtime, perkiraan biaya API
[Sel 2 — Code]      %load_ext autoreload / %autoreload 2
                    import sys; sys.path.append("../src")
[Sel 3 — Code]      cfg = load_config("../configs/base.yaml")
                    set_seed(cfg.seed)
[Sel 4..n]          Eksekusi, diselingi markdown penjelas
[Sel akhir-2]       ASSERT: validasi keluaran (jumlah baris, tidak ada NaN,
                    skema kolom sesuai kontrak)
[Sel akhir-1]       Tulis keluaran ke disk
[Sel akhir]         Ringkasan: tabel/angka kunci yang akan dikutip ke naskah
```

Aturan tambahan:
- Notebook **harus bisa dijalankan Restart & Run All** dari atas ke bawah tanpa intervensi manual.
- Jangan menyimpan `GEMINI_API_KEY` di dalam notebook; baca dari `.env` via `python-dotenv`.
- Pasang `nbstripout` agar output sel tidak membanjiri riwayat Git — kecuali pada notebook 10 yang memang menyimpan gambar hasil.

---

## 4. Lingkungan & Dependensi

Target: Python 3.11, Lenovo Legion 5 Slim (Ryzen 7 7840HS, RTX 4060 8 GB VRAM, RAM 16 GB).

```
# Inti
jupyterlab, ipykernel, python-dotenv, pyyaml, pandas, numpy, tqdm

# Ingesti
requests, beautifulsoup4, pymupdf (fitz), pdfplumber

# Retrieval
sentence-transformers, torch (CUDA), faiss-cpu, rank-bm25, Sastrawi

# Generator & evaluasi
google-genai, ragas, datasets

# Statistik & visual
scipy, scikit-learn, statsmodels, matplotlib
```

**Pilihan model (semua muat di 8 GB VRAM, fp16):**

| Peran | Kandidat | Alasan |
|---|---|---|
| Embedding | `BAAI/bge-m3` atau `intfloat/multilingual-e5-large` | Multibahasa, kuat untuk Bahasa Indonesia, ±2 GB fp16 |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder multibahasa, ±1 GB fp16 |
| Generator | Gemini via API (`temperature=0`) | Sesuai proposal, inference-only |
| Judge | Gemini via API, **model/versi berbeda dari generator jika memungkinkan** | Menekan *self-enhancement bias* (lihat §9.4) |

Kunci versi semua paket di `requirements.txt` (`==`), dan catat versi model embedding/reranker/API di `results/environment.json` agar hasil dapat direproduksi.

---

## 5. Kontrak Data (Skema)

Skema adalah antarmuka antar-notebook. Sepakati **sebelum** menulis kode; perubahan skema di tengah jalan adalah sumber kekacauan terbesar pada proyek semacam ini.

### 5.1 `data/processed/documents.jsonl`
```json
{
  "doc_id": "lpmb-2026-bbk8-edaran",
  "title": "Edaran Pelaksanaan BBK 8",
  "source_type": "pdf",
  "source_url": "https://lpmb.unair.ac.id/...",
  "local_path": "data/raw/pdf/794KR_....pdf",
  "sha256": "…",
  "program": "BBK",
  "year": 2026,
  "n_pages": 14,
  "retrieved_at": "2026-08-21T09:00:00+07:00"
}
```

### 5.2 `data/processed/chunks.jsonl`
```json
{
  "chunk_id": "lpmb-2026-bbk8-edaran::p03::c002",
  "doc_id": "lpmb-2026-bbk8-edaran",
  "page_start": 3, "page_end": 3,
  "section_heading": "C. Persyaratan Peserta",
  "text": "…",
  "n_chars": 812, "n_tokens": 198,
  "chunk_strategy": "recursive_800_120",
  "is_table": false
}
```
`chunk_id` menyertakan `chunk_strategy` di metadata agar indeks dari strategi berbeda tidak pernah tertukar.

`is_table` (ditambahkan setelah notebook 02–03 berjalan, perluasan aditif —
lihat README §Status): `true` untuk chunk yang berasal dari satu tabel PDF
utuh, dirender ke Markdown lewat `find_tables()`/`Table.extract()` (bukan
`extract_text()`, yang terbukti melinearkan sel tabel berkolom banyak
secara acak lintas baris — lihat README, temuan Tabel 2.3). Chunk tabel
SELALU satu tabel = satu chunk, tidak pernah dipecah `recursive_chunk()`
maupun digabung dengan paragraf lain, dan dikecualikan dari batas
`chunk_size` (atas) maupun `min_chunk_chars` (bawah) yang berlaku untuk
chunk prosa.

### 5.3 `data/benchmark/benchmark.jsonl` — **artefak kontribusi utama**
```json
{
  "qid": "Q-042",
  "question": "Kapan batas akhir unggah proposal Pengmas Mandiri 2026?",
  "category": "faktual_numerik",
  "is_out_of_scope": false,
  "answer_gold": "14 Maret 2026 pukul 23.59 WIB.",
  "evidence": [
    {"doc_id": "lpmb-2026-pengmas-mandiri", "page": 4,
     "span_text": "batas akhir unggah proposal adalah 14 Maret 2026"}
  ],
  "difficulty": "sedang",
  "author": "A1", "reviewer": "A2",
  "created_at": "2026-10-02"
}
```
`category` ∈ `{faktual_numerik, prosedural, dokumen_template, disambiguasi_program, luar_cakupan}` (sesuai subbab 3.3 proposal).
Untuk `is_out_of_scope: true` → `evidence` kosong dan `answer_gold` berisi pernyataan menolak menjawab.

### 5.4 `runs/{cond}/predictions.jsonl`
```json
{
  "qid": "Q-042", "condition": "cond_c",
  "retrieved": [{"chunk_id": "…", "rank": 1, "score_dense": 0.81,
                 "score_bm25": 12.4, "score_rrf": 0.032, "score_rerank": 0.94}],
  "answer": "Batas akhir unggah proposal adalah 14 Maret 2026 …",
  "abstained": false,
  "citations": ["lpmb-2026-pengmas-mandiri::p04::c001"],
  "latency_ms": {"retrieval": 42, "rerank": 310, "generation": 1840, "total": 2192},
  "tokens": {"prompt": 2140, "completion": 96},
  "cost_usd": 0.00087,
  "model": "gemini-…", "run_id": "2026-11-03T14:02:11", "cache_hit": false
}
```

### 5.5 `results/human_ratings.csv`
```
sample_id, annotator_id, faithfulness, relevance, correctness, abstain_correct, notes
```
`sample_id` adalah **kode buta**; peta `sample_id → (qid, condition)` disimpan terpisah di `results/blind_map.csv` dan baru dibuka pada notebook 09.

---

## 6. Rancangan Per-Notebook

### `00_setup_and_config.ipynb`
**Tujuan:** memastikan lingkungan siap sebelum tahap apa pun.
**Isi:** cek versi Python & CUDA (`torch.cuda.is_available()`), unduh & panggil sekali model embedding/reranker untuk verifikasi VRAM, uji satu panggilan Gemini API, buat seluruh direktori, tulis `results/environment.json`.
**Output:** `results/environment.json`, log verifikasi.
**Assert:** GPU terdeteksi; API mengembalikan respons; semua direktori ada.

---

### `01_ingest_documents.ipynb`
**Tujuan:** mengumpulkan bahan penelitian (tahapan 2 proposal).
**Langkah:**
1. Daftar URL sumber (Download Center, Informasi Penting, Materi & Template) — simpan sebagai `configs/sources.yaml`, bukan hardcode.
2. Unduh PDF → `data/raw/pdf/`, hitung `sha256`, hindari unduh ganda.
3. Snapshot HTML laman → `data/raw/web/` (arsipkan, karena laman berubah tiap periode).
4. Ekstraksi teks: PyMuPDF untuk teks umum, pdfplumber untuk halaman bertabel.
5. Isi metadata `program` dan `year` secara manual bila nama berkas kriptik (mis. "794KR PPENGMAS PKM DPR 2026") — sediakan sel tabel editable.

**Output:** `data/interim/*.txt`, `data/processed/documents.jsonl`
**Catatan penting:** dokumen yang hanya berupa gambar/poster **dicatat tetapi ditandai** `"extractable": false` — sesuai Batasan Masalah nomor 2. Laporkan jumlahnya; angka ini berguna sebagai temuan di naskah.

---

### `02_preprocess_chunking.ipynb`
**Tujuan:** menyiapkan basis pengetahuan.
**Langkah:** pembersihan header/footer berulang & nomor halaman; normalisasi whitespace; deteksi heading (regex "BAB", "A.", "1.") untuk mengisi `section_heading`; chunking rekursif dengan hormat pada batas paragraf; simpan `page_start/page_end` (wajib untuk pemetaan evidence).
**Parameter yang harus dieksplisitkan di config:** `chunk_size=800`, `chunk_overlap=120` (satuan karakter atau token — tetapkan satu dan konsisten).
**Output:** `data/processed/chunks.jsonl`
**Sel diagnostik:** histogram panjang chunk, jumlah chunk per dokumen, contoh 3 chunk acak untuk inspeksi mata. Chunk yang terlalu pendek (< 100 karakter) biasanya sampah hasil ekstraksi — buang dan laporkan jumlahnya.

---

### `03_build_indexes.ipynb`
**Tujuan:** membangun dua indeks yang dipakai kondisi B dan C.
**Langkah:**
1. Encode seluruh chunk dengan model embedding (batch, fp16, GPU) → simpan matriks `.npy`.
2. Bangun indeks FAISS (`IndexFlatIP` + normalisasi L2 sudah cukup untuk skala ribuan chunk; jangan pakai IVF/HNSW pada korpus sekecil ini — hanya menambah variabel pengganggu).
3. Bangun indeks BM25: tokenisasi + *stopword removal* + stemming Bahasa Indonesia (Sastrawi). **Simpan tokenizer yang sama** untuk dipakai saat query.
4. Uji sanity: kueri contoh, tampilkan top-5 dari masing-masing jalur berdampingan.

**Output:** `indexes/faiss/`, `indexes/bm25/`, `results/indexing_stats.json`
**Catatan:** waktu pengindeksan dan ukuran indeks dicatat — ini bagian dari analisis biaya (rumusan masalah 4).

---

### `04_build_benchmark.ipynb`
**Tujuan:** menyusun benchmark 150–250 pasang QA (kontribusi 1).
**Perancangan alur kerja anotasi:**
1. **Sel perencanaan kuota:** tabel target per kategori, misal faktual-numerik 60, prosedural 50, dokumen/template 30, disambiguasi 30, luar cakupan 30. Notebook menampilkan progres aktual vs target agar distribusi tidak timpang.
2. **Penulisan pertanyaan** dilakukan di luar notebook (spreadsheet/CSV) — lebih nyaman bagi anotator. Notebook bertugas **memuat, memvalidasi, dan mengunci**.
3. **Validasi otomatis (sel assert):**
   - `qid` unik; tidak ada pertanyaan duplikat (cek kemiripan kosinus > 0.95 → tandai untuk ditinjau);
   - setiap `span_text` **benar-benar ditemukan** di dalam teks dokumen pada halaman yang diklaim (pencocokan fuzzy, rasio ≥ 0.9). Ini menangkap salah ketik evidence sejak dini;
   - item `luar_cakupan` tidak memiliki evidence;
   - distribusi kategori sesuai target.
4. **Pemetaan evidence → chunk** (`evidence_mapping.py`): untuk setiap span, temukan chunk yang memuatnya → hasilkan `relevant_chunk_ids`. Disimpan sebagai file **turunan** `data/benchmark/qrels.jsonl`, bukan di dalam `benchmark.jsonl` (P2).
5. **Kesepakatan antar-anotator tahap dataset:** sampel ±20% item ditinjau anotator kedua; hitung kesepakatan pada label `category` dan pada ketepatan evidence.

**Output:** `data/benchmark/benchmark.jsonl`, `data/benchmark/qrels.jsonl`, `data/benchmark/annotation_log.csv`

---

### `05_run_experiments.ipynb` — **jantung ablation (P1)**
**Tujuan:** menjalankan ketiga kondisi lewat satu jalur kode.

Rancangan `src/pipeline.py`:
```python
def run_pipeline(question: str, cfg: PipelineConfig) -> Prediction:
    ctx = []
    if cfg.retrieval.mode != "none":
        cands = dense_search(question, k=cfg.retrieval.top_k_dense)
        if cfg.retrieval.mode == "hybrid":
            cands = rrf_fuse(cands, bm25_search(question, k=cfg.retrieval.top_k_bm25))
        if cfg.rerank.enabled:
            cands = cross_encoder_rerank(question, cands, top_n=cfg.rerank.top_n)
        ctx = cands[: cfg.retrieval.top_k_context]
    return generate(question, ctx, cfg.generator)   # satu prompt template
```
Notebook 05 hanya melakukan: muat 3 config → loop 3 kondisi × N pertanyaan → tulis `predictions.jsonl`.

**Isi `configs/cond_*.yaml`:**

| Kunci | cond_a | cond_b | cond_c |
|---|---|---|---|
| `retrieval.mode` | `none` | `dense` | `hybrid` |
| `retrieval.top_k_dense` | – | 5 | 20 |
| `retrieval.top_k_bm25` | – | – | 20 |
| `rerank.enabled` | false | false | true |
| `rerank.top_n` | – | – | 5 |
| `retrieval.top_k_context` | 0 | 5 | 5 |
| `generator.*` | identik | identik | identik |

**Prompt & abstain:** gunakan *structured output* Gemini dengan skema
`{"abstain": bool, "answer": str, "citations": [chunk_id]}`.
Deteksi abstain berbasis boolean jauh lebih andal daripada mencocokkan frasa "maaf, saya tidak menemukan…" dengan regex. Prompt sistem identik untuk B dan C; untuk A, blok konteks dihilangkan tetapi instruksi abstain tetap ada — jika tidak, kondisi A dihukum secara tidak adil pada metrik abstain dan ablation-nya cacat.

**Pengukuran:** catat latensi per tahap (retrieval / rerank / generasi terpisah), token, dan estimasi biaya. `cache_hit: true` **dikecualikan** dari statistik latensi — ini kesalahan yang mudah terjadi dan membuat angka latensi tidak bermakna.

**Output:** `runs/cond_{a,b,c}/predictions.jsonl`

---

### `06_eval_retrieval.ipynb`
**Tujuan:** metrik deterministik (hanya berlaku untuk kondisi B dan C).
**Metrik:** `recall@k` (k = 1, 3, 5, 10), `MRR`, `nDCG@10`, dihitung terhadap `qrels.jsonl`; dipecah per kategori pertanyaan.
**Hipotesis yang layak diuji di sini:** hybrid+rerank unggul terutama pada kategori `disambiguasi_program` dan `dokumen_template` (kueri dengan istilah eksak seperti nama program/kode dokumen) — persis argumen pada subbab 2.3.3. Tabel per-kategori inilah buktinya.
**Output:** `results/retrieval_metrics.csv`, `results/retrieval_by_category.csv`

---

### `07_eval_generation_ragas.ipynb`
**Tujuan:** metrik kualitas jawaban otomatis.
**Metrik:** `faithfulness`, `answer_relevance`, `answer_correctness` (RAGAS, judge = Gemini) untuk pertanyaan dalam cakupan.
**Metrik abstain (dihitung terpisah, bukan lewat RAGAS):**
- `abstain_recall` = proporsi item luar-cakupan yang benar ditolak;
- `false_answer_rate` = proporsi item luar-cakupan yang tetap dijawab (**metrik keselamatan paling penting** pada domain ini);
- `over_refusal_rate` = proporsi item dalam-cakupan yang justru ditolak.

**Catatan perancangan:** `faithfulness` tidak terdefinisi ketika konteks kosong. Untuk kondisi A, laporkan sel sebagai `n/a`, jangan diisi 0 — mengisi 0 akan menyesatkan pembacaan tabel hasil.
**Output:** `results/generation_metrics.csv`, `results/abstain_metrics.csv`

---

### `08_human_annotation_kit.ipynb`
**Tujuan:** menyiapkan paket penilaian manusia yang buta dan siap edar.
**Langkah:**
1. Sampling berstrata: ambil n ≈ 60–80 pertanyaan × 3 kondisi (≈ 180–240 unit penilaian), proporsional per kategori. Tetapkan `random_state`.
2. **Blinding:** acak urutan, ganti identitas kondisi dengan `sample_id` acak, sembunyikan skor otomatis.
3. Ekspor dua berkas identik (satu per anotator): `annotation_A1.csv`, `annotation_A2.csv` berisi kolom pertanyaan, jawaban sistem, konteks yang diambil, dan kolom skor kosong.
4. **Rubrik** ditulis sebagai markdown di notebook dan diekspor ke `docs/rubrik_penilaian.md`. Skala ordinal 1–4 (hindari 3 dan 5 yang mengundang jawaban tengah):

| Skor | Faithfulness | Correctness |
|---|---|---|
| 4 | Seluruh klaim didukung konteks | Sepenuhnya benar & lengkap |
| 3 | Klaim utama didukung, detail minor tak terdukung | Benar, ada kekurangan minor |
| 2 | Sebagian klaim tidak terdukung | Sebagian benar / menyesatkan sebagian |
| 1 | Klaim utama tidak terdukung (halusinasi) | Salah |

5. **Sesi kalibrasi:** 10 item dinilai bersama sebelum penilaian resmi, untuk menyamakan pemahaman rubrik. Wajib dicatat — ini yang membuat nilai kappa dapat dipertahankan saat sidang.

**Output:** `results/annotation_A1.csv`, `results/annotation_A2.csv`, `results/blind_map.csv` (jangan dibuka sampai notebook 09), `docs/rubrik_penilaian.md`

---

### `09_metric_validation.ipynb` — **inti kontribusi 3**
**Tujuan:** membuktikan (atau membantah) keandalan metrik otomatis untuk Bahasa Indonesia.
**Langkah:**
1. Muat kedua berkas anotasi; validasi kelengkapan.
2. **Kesepakatan antar-penilai:** karena rubrik ordinal 1–4, gunakan **Cohen's kappa berbobot kuadratik** (`weights="quadratic"`), dan laporkan Cohen's kappa polos sebagai pendamping. Sertakan interval kepercayaan via bootstrap. Argumennya: kappa tak berbobot memperlakukan selisih 1↔4 sama beratnya dengan 3↔4, padahal pada skala ordinal jelas tidak setara.
3. Jika kappa memadai (≥ 0.6), gabungkan penilaian dua anotator (rata-rata; untuk selisih ≥ 2 tingkat, adjudikasi manual).
4. **Buka blind map**, sandingkan skor manusia dengan skor RAGAS pada `sample_id` yang sama.
5. **Korelasi:** Pearson (linear) + Spearman (monotonik) per metrik (faithfulness, correctness, relevance), disertai CI bootstrap dan scatter plot.
6. **Analisis bias** (dari subbab 2.4.3): apakah galat judge berkorelasi dengan panjang jawaban (*verbosity bias*)? Regresikan selisih (skor otomatis − skor manusia) terhadap jumlah token jawaban.
7. **Analisis kegagalan:** ambil 10 item dengan selisih terbesar, baca manual, kategorikan pola kesalahan judge. Bagian kualitatif ini biasanya menjadi paragraf paling bernas di Bab IV.

**Output:** `results/agreement.csv`, `results/correlation.csv`, `figures/scatter_auto_vs_human.png`

---

### `10_analysis_and_figures.ipynb`
**Tujuan:** menghasilkan seluruh tabel & gambar untuk Bab IV.
**Isi:**
- Tabel utama: 3 kondisi × seluruh metrik.
- **Uji signifikansi:** karena ketiga kondisi dievaluasi pada pertanyaan yang sama, gunakan uji berpasangan (Wilcoxon signed-rank atau bootstrap berpasangan), **bukan** uji t independen. Tiga perbandingan pasangan → koreksi Holm. Laporkan juga *effect size*, karena signifikansi pada n≈200 tidak otomatis berarti besar.
- **Trade-off (rumusan masalah 4):** scatter akurasi vs latensi p50/p95, dan akurasi vs biaya per 100 kueri. Reranking cross-encoder adalah komponen paling mahal secara latensi — kuantifikasi berapa poin akurasi yang dibeli dengan berapa milidetik.
- Ekspor semua gambar 300 dpi ke `figures/`.

**Output:** `results/main_results.csv`, `results/significance_tests.csv`, `figures/*.png`

---

### `11_prototype_app.ipynb`
**Tujuan:** artefak praktis (Manfaat Penelitian nomor 4), skala minimum.
**Isi:** UI percakapan sederhana (`ipywidgets` di dalam notebook, atau skrip Gradio/Streamlit yang diluncurkan dari notebook) yang memanggil `run_pipeline` dengan `cond_c`, menampilkan jawaban + rujukan dokumen + tautan halaman sumber, dan menampilkan pesan menolak menjawab bila di luar cakupan. Cukup untuk memenuhi Gambar 3.3 — jangan berlebihan; kontribusi skripsi ini ada di data dan evaluasi.

---

## 7. Peta Notebook → Tahapan & Jadwal Penelitian

| Bulan (dari Agustus 2026) | Tahapan proposal (subbab 3.3) | Notebook | Milestone terverifikasi |
|---|---|---|---|
| 1 | Studi literatur | — | Rancangan ini disetujui pembimbing |
| 1–2 | Pengumpulan & pra-pemrosesan | 00, 01, 02 | `chunks.jsonl` terbentuk & terinspeksi |
| 2–3 | Penyusunan benchmark | 04 (+03) | 150–250 QA lolos seluruh assert |
| 3–4 | Implementasi 3 strategi | 03, 05 | 3 kondisi berjalan pada 20 pertanyaan uji |
| 4–5 | Eksperimen & pengukuran | 05, 06, 07 | `predictions.jsonl` lengkap untuk 3 kondisi |
| 5–6 | Validasi metrik & penilaian manusia | 08, 09 | Kappa & korelasi terhitung |
| 6 | Analisis & penyusunan laporan | 10, 11 | Seluruh tabel/gambar Bab IV siap |

**Peringatan jadwal:** notebook 04 (penyusunan benchmark) hampir selalu memakan waktu lebih lama dari perkiraan karena bersifat manual dan bergantung pada ketersediaan anotator kedua. Kerjakan notebook 03 dan kerangka 05 secara paralel pada bulan 2–3 agar bulan 3–4 tidak menumpuk.

---

## 8. Reproduktibilitas & Pengendalian Biaya

1. **Cache LLM (`src/llm_cache.py`).** Kunci = `sha256(model + prompt + parameter generasi)`, nilai = respons + metadata. Disimpan di SQLite. Efeknya: menjalankan ulang notebook 05/07 nyaris gratis, dan hasil eksperimen stabil antar-eksekusi.
2. **Uji kering (dry run).** Setiap notebook yang memanggil API punya flag `cfg.limit = 20` di sel config. Kembangkan dengan 20 pertanyaan; baru set `None` untuk eksekusi penuh.
3. **`temperature = 0`** untuk generator maupun judge; catat `seed` bila API mendukung.
4. **Anggaran API.** Sebelum eksekusi penuh, estimasi biaya: `3 kondisi × N pertanyaan × token rata-rata`, ditambah panggilan RAGAS (RAGAS memanggil judge beberapa kali per item — biasanya komponen biaya terbesar, sering di luar dugaan). Tampilkan estimasi di sel sebelum loop dijalankan.
5. **`run_id`** pada setiap baris prediksi, agar eksekusi lama tidak tertimpa diam-diam.
6. Git: commit notebook yang sudah di-strip; `data/raw/`, `cache/`, dan `.env` masuk `.gitignore`; sertakan `configs/sources.yaml` agar orang lain dapat mengunduh ulang dokumen yang sama.

---

## 9. Risiko Teknis & Mitigasi

| Risiko | Dampak | Mitigasi (sudah terancang di atas) |
|---|---|---|
| Ground-truth terikat `chunk_id`, lalu chunking diubah | Benchmark hangus, kerja anotasi terbuang | Evidence berbasis span; pemetaan ke chunk sebagai turunan (§5.3, notebook 04) |
| Kode pipeline disalin ke 3 notebook, lalu tidak sinkron | Klaim ablation runtuh | Satu `run_pipeline`, dibedakan config (§6, notebook 05) |
| Gemini menjadi generator **dan** judge | *Self-enhancement bias*, temuan mudah diserang penguji | Gunakan model/versi judge yang berbeda bila tersedia; jika tidak, nyatakan eksplisit sebagai keterbatasan dan sandarkan kesimpulan pada korelasi terhadap manusia — yang justru merupakan tujuan penelitian ini |
| Kappa rendah (< 0.4) | Kriteria validasi tidak layak dipakai | Sesi kalibrasi + rubrik dengan jangkar konkret (notebook 08); bila tetap rendah, itu **temuan sah** — laporkan dan analisis penyebabnya |
| Ekstraksi PDF berantakan (tabel, dua kolom) | Chunk sampah → retrieval buruk | pdfplumber untuk halaman bertabel; sel inspeksi manual & pembuangan chunk terlalu pendek (notebook 02) |
| Kuota/biaya API terlampaui di tengah eksperimen | Eksperimen tidak selesai | Cache + dry run + estimasi biaya pra-eksekusi (§8) |
| Anotator kedua sulit dijadwalkan | Bab validasi tertunda | Paket anotasi mandiri (CSV + rubrik) disiapkan lebih awal di notebook 08 |
| Laman LPMB berubah di tengah penelitian | Dokumen tidak konsisten dengan benchmark | Snapshot + `sha256` + `retrieved_at`; bekukan korpus pada satu tanggal dan nyatakan di naskah |

---

## 10. Checklist Kesiapan (Definition of Done)

- [ ] 12 notebook dapat dijalankan **Restart & Run All** berurutan tanpa galat
- [ ] Seluruh sel assert lolos
- [ ] `benchmark.jsonl` berisi 150–250 item; distribusi kategori terpenuhi; setiap evidence terverifikasi ada di dokumen
- [ ] `predictions.jsonl` lengkap untuk 3 kondisi, tanpa `qid` yang hilang
- [ ] Cohen's kappa berbobot terhitung beserta CI
- [ ] Korelasi otomatis–manusia terhitung beserta CI dan scatter plot
- [ ] Uji signifikansi berpasangan + koreksi Holm terlaporkan
- [ ] Angka latensi & biaya tercatat, sudah mengecualikan `cache_hit`
- [ ] `requirements.txt` terkunci; `environment.json` tersimpan
- [ ] Tidak ada kunci API di dalam notebook mana pun
- [ ] `figures/` memuat seluruh gambar Bab IV pada 300 dpi

---

## 11. Urutan Pengerjaan yang Disarankan

Bangun **rangka ujung-ke-ujung terlebih dahulu** dengan 10 pertanyaan contoh dan 5 dokumen — notebook 01 → 05 → 06 dalam bentuk paling sederhana. Setelah satu angka evaluasi berhasil keluar dari ujung pipeline, barulah perbesar skala (dokumen lengkap, benchmark penuh, tiga kondisi).

Alasannya: kesalahan perancangan kontrak data (§5) hampir selalu baru terlihat ketika data mengalir sampai ujung. Menemukannya pada minggu kedua jauh lebih murah daripada menemukannya setelah 200 pertanyaan selesai dianotasi.
