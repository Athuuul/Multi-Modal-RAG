# 📘 Multi-Modal RAG for IMF Qatar Staff Report

**Text + Tables + Charts + Images | FAISS + CLIP + ColPali + Qwen2-VL**

This repository contains a **production-grade Multi-Modal Retrieval-Augmented Generation (RAG)** system capable of understanding complex PDF documents containing:

* Free-form **text**
* **Tables** (Camelot)
* **Charts / Figures** (detected via edge heuristics)
* **Full-page images**
* **OCR output** for scanned pages

It supports **FAISS** vector search, **CLIP** image embeddings, **MiniLM** text embeddings, **ColPali** page retrieval, and **Qwen2-VL** multimodal answer generation. A full **Streamlit UI** is included for interactive QA.

---

## 🚀 Features

### 🔹 Multi-modal PDF ingestion

* Extracts text with `pdfplumber`
* Extracts tables with `camelot`
* Detects figures using structural edge density
* OCR extraction using `pytesseract`
* Converts each PDF page to high-resolution JPEG

### 🔹 Dual embedding pipeline

* **Text:** `sentence-transformers/all-MiniLM-L6-v2`
* **Images:** `openai/clip-vit-base-patch32`
* All vectors padded to a unified FAISS dimension

### 🔹 Unified FAISS vector index

* Stores **text**, **tables**, **chart captions**, and **images**
* Uses `IndexFlatIP` for cosine-like similarity
* Metadata includes: modality, page number, doc ID, raw content

### 🔹 Vision-based retrieval (ColPali)

* Retrieves similar PDF pages based on page images
* Merged with FAISS results

### 🔹 RAG answer generator

* **Qwen2-VL-2B** multimodal LLM
* Citations enforced via `[p.X]`
* Text-only fallback mode for low-memory GPUs

### 🔹 Streamlit Web UI

* Adjustable retrieval parameters
* Displays retrieved contexts and page images
* Generates final answer with citations
* Works in **Colab using ngrok**

### 🔹 Evaluation Suite

* Containment score
* Exact match
* Per-modality accuracy (text/table/chart)

---

## 📂 Project Structure

```
Multi-Modal-RAG/
│
├── data/
│   └── QATAR_DOC.pdf
│
├── multimodal_index.faiss
├── metadata_store.pkl
│
├── streamlit_app.py
├── rag_backend.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Athuuul/Multi-Modal-RAG.git
cd Multi-Modal-RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your PDFs

Place PDF files inside:

```
data/
```

---

## ▶️ Running the Streamlit App (Local Machine)

```bash
streamlit run streamlit_app.py
```

---

## ▶️ Running in Google Colab (with ngrok)

Install:

```bash
!pip install streamlit pyngrok
```

Authenticate ngrok:

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
```

Run Streamlit in background:

```bash
!nohup streamlit run streamlit_app.py --server.port 8501 --server.headless true &
```

Open tunnel:

```python
public_url = ngrok.connect(8501)
public_url
```

---

## 🧠 Pipeline Overview

### 1. Ingestion

PDF → text, tables, charts, OCR → chunk metadata.

### 2. Embeddings

* Text → MiniLM
* Images → CLIP
* Normalized + padded

### 3. Indexing

* Stored inside FAISS (IP/cosine)

### 4. Retrieval

Combined:

* FAISS top-K
* ColPali top-K pages
* Dedup + reweight

### 5. Context Assembly

Select best text + images for the LLM.

### 6. Answer Generation

Qwen2-VL produces a grounded answer with `[p.X]` citations.

---

## 🧪 Evaluation

```python
from rag_backend import run_evaluation
results = run_evaluation()
print(results)
```

---

## 📋 Requirements

```
transformers==4.45.0
sentence-transformers
byaldi
pdfplumber
camelot-py
pytesseract
faiss-cpu
opencv-python-headless
pdf2image
torch
streamlit
matplotlib
qwen-vl-utils
```

---

## 🔮 Future Improvements

* Reranking (bge-reranker, ColBERT)
* HNSW or IVF FAISS index for speed
* Replace Tesseract with PaddleOCR
* Deploy automatically using GitHub Actions
* HuggingFace Spaces / Cloudflare app

---

## 📜 License

MIT License.
