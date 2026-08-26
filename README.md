# Skripsi RAG LPMB UNAIR

Evaluasi Komparatif Strategi RAG (Closed-Book, Naive RAG, Hybrid + Reranking)
serta Validasi Metrik LLM-as-Judge untuk Sistem Tanya-Jawab Berbahasa
Indonesia. Rancangan teknis lengkap ada di [`perancangan.md`](perancangan.md)
— baca dokumen itu dulu sebelum menambah notebook atau modul baru.

## Status

Scaffold: struktur direktori, konfigurasi (`configs/`), dan stub modul
(`src/*.py`, signature + docstring kontrak, badan fungsi `NotImplementedError`
kecuali yang disebut di bawah) sudah ada.

- `src/config.py` — fungsional.
- `notebooks/00_setup_and_config.ipynb` — fungsional.
- `src/ingest.py` — fungsional (download/snapshot/ekstraksi PyMuPDF+pdfplumber/doc_id/JSONL).
- `notebooks/01_ingest_documents.ipynb` — fungsional, sudah dijalankan
  end-to-end terhadap PDF di `data/raw/pdf/` → `data/processed/documents.jsonl`.
  Alur saat ini mengasumsikan PDF sudah ditaruh manual di `data/raw/pdf/`
  (bukan diunduh dari `configs/sources.yaml`); fungsi unduh di `src/ingest.py`
  tetap tersedia untuk dipakai belakangan bila korpus diperluas dari web.
  **Ekstraksi web (baru):** `snapshot_web_page` kini juga menulis sidecar
  `{stem}.meta.json` (url asli + `retrieved_at`) di sebelah berkas HTML —
  tanpa ini url sumber tidak bisa dipulihkan lagi dari nama berkas (slug
  dipotong/diubah). `extract_text_html` (BeautifulSoup) membuang
  `script`/`style`/`nav`/`header`/`footer`/`noscript` lalu mengembalikan
  teks laman dibungkus sebagai satu "halaman" (`page=1`, sama skema
  `list[{"page","text",...}]` dengan jalur PDF) supaya
  `write_interim_text`/`is_extractable`/`build_document_record` dipakai
  ulang tanpa cabang kode terpisah. Notebook 01 sekarang punya seksi
  paralel untuk `data/raw/web/*.html` (baru men-scan snapshot yang sudah
  ada, tidak mengunduh), dan perakitan record PDF+web digabung lewat
  fungsi baru `ingest.assemble_documents()` (dipanggil dua kali, sekali
  per sumber, `existing_ids` yang sama supaya `doc_id` tetap unik lintas
  sumber) alih-alih loop terpisah yang identik untuk tiap sumber.
  Diverifikasi end-to-end (`nbconvert --execute`) dengan 0 snapshot HTML
  di korpus contoh saat ini (graceful, tidak mengganggu `documents.jsonl`
  yang sudah ada) dan dites terpisah dengan satu sample HTML tiruan
  (nav/script/footer terbuang, `source_url` terisi dari sidecar).
  **Ekstraksi tabel (revisi):** semula hanya memakai `page.extract_text()`
  pdfplumber untuk halaman bertabel, dengan alasan `extract_tables()` mentah
  menghasilkan noise `"| | | |"` pada tabel berkolom banyak-kosong. Ternyata
  `extract_text()` punya cacat yang JAUH lebih serius: pada tabel berkolom
  banyak ia melinearkan sel per POSISI-BACA bukan per BARIS, sehingga nilai
  antar-kolom (nama skema, jumlah anggota, nominal dana) tercampur lintas
  baris — persis kategori `faktual_numerik` benchmark bergantung pada data
  ini. Sekarang: `find_tables()` + `Table.extract()` dipakai untuk
  merender tiap tabel ke Markdown per-baris (`_rows_to_markdown`), dan
  bbox tabel yang lolos difilter dari teks naratif (`page.filter()`)
  supaya tidak terduplikasi. Tabel "kandidat" dengan < 2 kolom nyata atau
  tanpa baris data (`find_tables()` kerap salah mendeteksi sel header
  berbaris-banyak sebagai tabel 1-kolom bersarang tersendiri — lihat kasus
  "Mitra Sasaran" hal. 32 dokumen contoh) DITOLAK sebagai tabel, dan
  bbox-nya SENGAJA tidak dikecualikan dari teks naratif, supaya kontennya
  tetap muncul lewat extract_text() biasa alih-alih hilang.
- `src/chunking.py` — fungsional, chunking *structure-aware*: heading `BAB`
  = batas keras, sub-heading (`A.`, `2.1.`, dst) = metadata `section_heading`
  saja (batas lunak). Paragraf > soft_max dipecah via kaskade
  paragraf→kalimat→kata→karakter dengan overlap adaptif (tidak pernah
  melanggar chunk_size pada sub-piece hasil pecahnya, beda dari overlap
  naif yang bisa menduplikasi teks pada batas potongan besar — lihat
  komentar di `_pack_pieces`; lihat juga catatan `soft_max` di bawah).
  **Bug yang pernah ditemukan & diperbaiki:** overlap antar-chunk
  (`recursive_chunk` maupun `_pack_pieces`) awalnya memotong `chunk_overlap`
  karakter mentah dari ekor teks (`text[-n:]`) tanpa memperhatikan batas
  kata — pada korpus nyata ini memenggal kata di 158/256 chunk (mis.
  "Pelaksana" → "elaksana"). Bukan disebabkan oleh normalisasi whitespace
  (`clean_text` sudah benar sejak awal, mempertahankan `\n`), tapi oleh
  overlap berbasis karakter yang tidak snap ke spasi. Diperbaiki dengan
  `_word_boundary_tail()` (membuang kata/paragraf parsial di awal potongan
  overlap, walau artinya overlap kadang lebih pendek dari `chunk_overlap`).
  Sekaligus, `_pack_pieces` diubah agar menggabungkan paragraf berbeda
  dalam satu chunk dengan `"\n\n"` (bukan spasi tunggal), supaya batas
  paragraf tidak hilang saat >1 paragraf dipaketkan ke satu chunk.
  **Bug daftar isi (TOC) — ditemukan & diperbaiki:** halaman TOC tidak
  punya baris kosong antar-entri, jadi `_split_paragraph_blocks` (yang
  memecah berdasar `\n\s*\n`) menggabungkan SELURUH halaman TOC jadi satu
  "paragraf" raksasa; `_cascade_split` lalu jatuh ke pemecahan
  kalimat/kata (bukan struktur asli), menghasilkan chunk ~700-800 karakter
  isi dot leader murni (mis. `"HALAMAN JUDUL ................. i"`) yang
  murni derau retrieval. Ini SEKALIGUS menjelaskan dua gejala lain: (a)
  pola nomor entri TOC ("2.2. Program Pengabdian ... 5") kebetulan cocok
  regex heading yang sama dengan heading sungguhan ("2.1. Pendahuluan"),
  jadi `section_heading` ikut terisi baris TOC lengkap dengan titik-titik;
  (b) chunk yang dipaketkan dari fragmen kalimat TOC yang pendek-pendek
  rapat mengisi hingga mendekati `chunk_size`, menyamarkan seolah-olah
  *structure-aware chunking* tidak berjalan sama sekali (hard cap di
  ~800 untuk hampir semua chunk) — padahal begitu chunk TOC dibuang,
  distribusi ukuran chunk non-tabel kembali bervariasi wajar (109-799,
  median ~620, bukan menumpuk di batas atas).
  Perbaikan dua bagian: (1) `detect_section_heading` menolak baris yang
  mengandung dot leader (`\.{3,}`, ≥3 titik beruntun) sebelum dicocokkan
  ke `_HEADING_PATTERNS`; (2) `build_chunk_records` membuang chunk
  non-tabel dengan rasio karakter `.` di atas `max_dot_ratio` (baru,
  `configs/base.yaml::chunking.max_dot_ratio`, default 0.20 — dipilih
  empiris: chunk TOC di korpus contoh berkisar 0.24-0.88, sedangkan chunk
  prosa asli tertinggi [termasuk lampiran template berformat titik-titik
  isian] cuma 0.10, ada celah lebar). Chunk tabel dikecualikan dari kedua
  perbaikan ini (sel tabel/kode dokumen wajar mengandung banyak titik).
  **Chunk tabel (field baru `is_table`):** blok tabel Markdown yang
  disisipkan `ingest.write_interim_text` (penanda
  `=== TABEL HALAMAN {n} #{j} === ... === AKHIR TABEL ===`) dipisahkan
  dari teks naratif SEBELUM `clean_text()` (supaya sintaks Markdown tidak
  kena heuristik pembersihan prosa), lalu jadi unit tersendiri yang SELALU
  dipaketkan sebagai satu piece `standalone` di `_units_to_pieces` — tidak
  pernah lewat `recursive_chunk()`, tidak pernah digabung dengan paragraf
  lain, dan (beda dari piece lain) DIKECUALIKAN dari batas atas
  `chunk_size` maupun batas bawah `min_chunk_chars`: satu tabel = satu
  chunk utuh, apa pun ukurannya. Setiap record chunk kini punya field
  `is_table: bool` (perluasan aditif atas skema §5.2 `perancangan.md` —
  konsumen lama yang cuma baca `text`/`chunk_id` tidak terpengaruh).
  **Kalibrasi ulang `chunk_size` (800→2000) + `soft_max` (baru, 2600) —
  rekomendasi eksternal, diverifikasi & diterapkan:** analisis distribusi
  ukuran "bagian" (unit logis antar-heading) di korpus contoh menunjukkan
  median ~826-1235 dan p75 ~1871-1980 tergantung metode pengukuran —
  chunk_size lama (800) memotong lebih dari separuh bagian; hanya ~47-48%
  bagian yang utuh dalam satu chunk. Menaikkan ke 2000 menaikkan itu ke
  ~78-81%; setelah 2000 hasilnya landai (2400 cuma +3-4 poin), jadi 2000
  dipilih sebagai titik yang tidak perlu menaikkan risiko chunk terlalu
  besar mengencerkan embedding. Konsep BARU `soft_max` (2600, ~1.3x
  chunk_size) memisahkan "target pemaketan" dari "batas paksa dipecah":
  paragraf tunggal sepanjang (chunk_size, soft_max] TIDAK lagi dipaksa
  lewat `recursive_chunk()` (pemotongan kalimat/kata yang mengorbankan
  koherensi semantik) — dibiarkan mengalir lewat `_pack_pieces` normal,
  yang secara alami mengeluarkannya sebagai satu chunk sendiri begitu
  sudah melebihi chunk_size (lengkap dengan overlap ke tetangga). Hanya
  paragraf > soft_max yang sungguh dipaksa dipecah.
  **Overlap kini unit-based, bukan char-slice:** `_pack_pieces` mengambil
  PARAGRAF TERAKHIR UTUH dari chunk sebelumnya sebagai overlap (bukan N
  karakter terakhir dari keseluruhan teks chunk, yang berpotensi memotong
  di tengah kalimat walau sudah snap ke batas kata) — hanya dipangkas ke
  batas kata (`_word_boundary_tail`) kalau paragraf itu sendiri > anggaran
  overlap (chunk_overlap, dinaikkan 120→200, tetap ~10% dari chunk_size).
  **Chunk "yatim" kini dilebur, bukan dibuang** (`min_chunk_chars`
  dinaikkan 100→250 supaya lebih banyak chunk pendek tertangkap sebagai
  kandidat lebur, bukan lolos sebagai chunk-mini yang tidak informatif):
  `_merge_orphan_chunks` — tiga lintasan (mundur ke chunk sebelumnya,
  lalu maju ke chunk berikutnya, lalu ke tabel tetangga sebagai
  preamble/caption), tidak pernah melebur lintas batas BAB maupun ke/dari
  chunk lain yang statusnya tabel-tapi-bukan-target-lebur-tabel. Temuan
  saat implementasi: SEMUA chunk yatim di korpus contoh (15/15) ternyata
  terjepit di antara dua chunk tabel (caption tabel pendek sebelum tabel
  berikutnya) — lintasan mundur/maju biasa (ke sesama non-tabel) tidak
  cukup, perlu lintasan ketiga yang melebur ke tabel karena struktur
  `_extract_units` menjamin narasi satu halaman selalu mendahului tabel
  halaman itu (jadi paragraf pendek tepat sebelum tabel hampir selalu
  caption/pengantar tabel itu, bukan derau lepas).
- `notebooks/02_preprocess_chunking.ipynb` — fungsional, dijalankan
  end-to-end → `data/processed/chunks.jsonl` (114 chunk dari 99 halaman
  setelah seluruh perbaikan di atas — 50 di antaranya chunk tabel, 6
  dibuang sebagai derau TOC, 15 chunk yatim dilebur bukan dibuang [0
  dibuang karena kependekan]. Ukuran chunk non-tabel: min 267, median
  1446, p75 1845, maks 2511 [semua ≤ soft_max 2600]. 98% chunk punya
  `section_heading` terisi; diverifikasi tidak ada lagi chunk yang mulai
  di tengah kata, tidak ada lagi tabel yang nilainya tercampur lintas
  baris atau terpotong di tengah [Tabel 2.2 & 2.3 dicek manual], tidak
  ada lagi halaman TOC yang ikut terindeks [rasio titik tertinggi yang
  tersisa 0.089, jauh di bawah ambang 0.20], dan bebas duplikasi lewat
  inspeksi manual). Sel ASSERT diperbarui: batas atas chunk non-tabel
  sekarang soft_max (bukan chunk_size), batas bawah min_chunk_chars
  sekarang juga dikecualikan untuk chunk tabel (celah yang baru
  ketahuan saat menaikkan min_chunk_chars — tabel kecil di korpus
  contoh bisa serendah 126 karakter).
- `src/indexing.py` — fungsional. `build_faiss_index`/`load_faiss_index`
  diberi parameter `chunk_ids` tambahan (tidak ada di stub awal) karena
  perlu menyimpan sendiri pemetaan posisi→chunk_id yang disebut di
  docstring-nya. `tokenize_id` memakai Sastrawi: stopword removal
  dulu baru stemming (kamus stopword Sastrawi sudah memuat banyak bentuk
  berimbuhan, jadi mengurangi kata yang perlu distem).
- `notebooks/03_build_indexes.ipynb` — fungsional, dijalankan end-to-end
  terhadap `data/processed/chunks.jsonl` → `indexes/faiss/`,
  `indexes/bm25/`, `results/indexing_stats.json` (114 chunk setelah
  seluruh perbaikan chunking & ekstraksi tabel di atas, embedding
  BAAI/bge-m3 dim 1024, index FAISS <500 KB, index BM25 ~117 KB; encode
  ~40 detik, build BM25 ~77 detik didominasi stemming Sastrawi per-chunk).
  Sanity check kueri contoh menunjukkan top hit yang sama antara jalur
  dense dan BM25 pada dokumen yang relevan.
  **Catatan path:** notebook menulis output pakai path literal `"../indexes/..."`
  dkk, BUKAN `cfg.paths.*` langsung — `cfg.paths` relatif ke root proyek
  sedangkan cwd kernel notebook ada di `notebooks/`, jadi `Path(cfg.paths.x)`
  akan salah tempat kecuali diberi prefix `"../"` (ikuti pola notebook 01/02).

Notebook 04–11 belum dibuat — lihat §6 `perancangan.md` untuk rancangan
tiap notebook secara berurutan. Notebook 04 (build benchmark QA) adalah
langkah berikutnya, dan mengonsumsi `data/processed/chunks.jsonl` +
`indexes/`.

## Mulai

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # lalu isi GEMINI_API_KEY
jupyter lab
```

Jalankan `notebooks/00_setup_and_config.ipynb` (Restart & Run All) untuk
memverifikasi lingkungan: Python/CUDA, model embedding & reranker termuat,
satu panggilan Gemini API berhasil, seluruh direktori keluaran ada.

## Struktur

Lihat §2 `perancangan.md` untuk struktur direktori lengkap dan §5 untuk
kontrak skema data (`documents.jsonl`, `chunks.jsonl`, `benchmark.jsonl`,
`predictions.jsonl`, `human_ratings.csv`) — skema ini adalah antarmuka
antar-notebook dan tidak boleh diubah secara ad hoc.

## Prinsip yang wajib dijaga

- **P1** — ketiga kondisi ablation (A/B/C) HARUS lewat `src/pipeline.py::run_pipeline`
  yang sama, dibedakan hanya lewat `configs/cond_*.yaml`.
- **P2** — ground-truth benchmark diikat ke `(doc_id, halaman, span_text)`,
  bukan ke `chunk_id`.
- **P3** — semua panggilan Gemini API lewat `src/llm_cache.py`.
- **P4** — logika inti di `src/*.py`; notebook memanggil & menarasikan.
- **P5** — penilaian manusia dikumpulkan buta terhadap kondisi dan skor
  otomatis (lihat notebook 08).
