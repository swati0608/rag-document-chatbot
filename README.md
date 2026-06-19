---
title: RAG Document Chatbot
emoji: 📄
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 📄 RAG Document Chatbot

A production-quality **Retrieval-Augmented Generation** chatbot. Upload a PDF or TXT file and ask grounded questions about its contents — answers cite the exact passages they came from.

> Built with open-source components only. **No OpenAI key required.**

---

## ✨ Features

- 📤 **Upload PDF or TXT** — parsed with PyMuPDF for fast, accurate text extraction
- ✂️ **Smart chunking** — `RecursiveCharacterTextSplitter` (size 500, overlap 50)
- 🧠 **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2`, runs free on CPU
- 🗂️ **Vector search** — in-memory ChromaDB with cosine similarity (top-3 retrieval)
- 🤖 **Open-source LLM** — Mistral-7B-Instruct via HuggingFace Inference API
- 💬 **Chat with history** — keeps the last 5 exchanges visible
- 🔍 **Source transparency** — every answer links to the exact chunks it used
- 🧹 **One-click reset** for both conversation and index
- 🎨 **Clean dark theme**, two-column responsive layout, mobile-friendly
- 🛡️ **Robust error handling** for unsupported files, empty queries, and API failures

---

## 🖼️ Screenshots

> _Add screenshots here once deployed:_
> - `docs/screenshot-upload.png` — file upload + document info card
> - `docs/screenshot-chat.png` — chat with answer + sources expanded
> - `docs/screenshot-mobile.png` — mobile responsive view

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Gradio UI  │ ──► │  RAG Pipeline    │ ──► │ HuggingFace  │
│  (app.py)   │     │  (rag_pipeline)  │     │ Inference    │
└─────────────┘     └──────────────────┘     └──────────────┘
       │                    │     ▲
       │                    ▼     │
       │            ┌──────────────────┐
       └──────────► │  ChromaDB        │
                    │  + MiniLM-L6-v2  │
                    └──────────────────┘
```

| Layer | File | Responsibility |
|---|---|---|
| UI | `app.py` | Gradio Blocks app, event wiring, state |
| Parsing | `document_processor.py` | PDF/TXT → cleaned text |
| RAG | `rag_pipeline.py` | Chunk, embed, index, retrieve, prompt |
| LLM | `llm_handler.py` | HuggingFace API client with error handling |

---

## 🚀 Run locally

### 1. Clone & install

```bash
git clone <your-repo-url> rag-document-chatbot
cd rag-document-chatbot

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Get a free HuggingFace token

1. Go to **https://huggingface.co/join** and create a free account (takes ~30 seconds).
2. Visit **https://huggingface.co/settings/tokens**.
3. Click **"New token"**, give it any name (e.g. `rag-chatbot`), and select role **Read**.
4. Click **Generate token**, then copy the token (starts with `hf_…`).
5. Accept the model license at **https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3** (one-click, free).

### 3. Configure environment

```bash
cp .env.example .env
# then edit .env and paste your HF token
```

### 4. Run

```bash
python app.py
```

Open **http://localhost:7860** in your browser. Upload a PDF or TXT and start asking questions.

---

## ☁️ Deploy to HuggingFace Spaces

1. Create a new Space at **https://huggingface.co/new-space** → SDK: **Gradio**.
2. Push all files in this repo to the Space.
3. In **Space Settings → Variables and secrets**, add a secret:
   - Name: `HF_TOKEN`
   - Value: your `hf_…` token
4. The Space rebuilds automatically and your chatbot goes live.

The YAML metadata block at the top of this README is read by HF Spaces to configure the SDK and entry point.

---

## ⚙️ Configuration

All key parameters are constructor arguments — no magic numbers buried in code.

```python
# rag_pipeline.py
RAGPipeline(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50,
    top_k=3,
)

# llm_handler.py
HuggingFaceLLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=512,
    temperature=0.3,   # low = factual
    timeout=60,
)
```

---

## 🧰 Common errors & fixes

| Error | Cause | Fix |
|---|---|---|
| `HF_TOKEN not set` | `.env` missing or not loaded | Copy `.env.example` → `.env`, paste token, restart |
| `401 invalid or expired HF_TOKEN` | Token wrong / revoked | Generate a new token at huggingface.co/settings/tokens |
| `403` on model | Mistral license not accepted | Visit the model page and click "Agree and access" |
| `503 model is loading` | Cold start (first call after idle) | Wait 20–30s and ask again — this is normal on free tier |
| `429 / 402 quota exceeded` | HF free-tier rate limit | Wait a few minutes, or upgrade HF plan |
| `Unsupported file type` | File isn't `.pdf` or `.txt` | Convert to one of the supported formats |
| `EmptyDocumentError` | PDF is image-only (scanned) | Run OCR first (e.g. `ocrmypdf in.pdf out.pdf`) |
| Slow first run | Downloading MiniLM (~90 MB) | One-time; cached afterwards in `~/.cache/huggingface` |
| `Port 7860 already in use` | Another Gradio app running | `PORT=7861 python app.py` |
| Chroma telemetry warnings | Cosmetic | Already disabled via `anonymized_telemetry=False` |

---

## 🧪 Tech stack

- **UI:** Gradio 4
- **LLM:** Mistral-7B-Instruct-v0.3 (HF Inference Providers)
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2`
- **Vector store:** ChromaDB (in-memory)
- **Splitter:** `langchain-text-splitters` (`RecursiveCharacterTextSplitter`)
- **PDF:** PyMuPDF (`fitz`)
- **Runtime:** Python 3.10+

---

## 📜 License

MIT — feel free to use this as a starting point for client projects.
