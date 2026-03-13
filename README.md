# 🎙️ VoiceRAG — Voice-Enabled PDF Question Answering

A fully local, voice-activated RAG (Retrieval-Augmented Generation) system that lets you upload any PDF and ask questions using your voice.

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Ollama** — [Download](https://ollama.ai) (runs LLMs locally)
- **ffmpeg** — Required by Whisper for audio processing
  - Windows: `winget install ffmpeg` or [download](https://ffmpeg.org/download.html)
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

---

### 2. Clone / Open in VS Code

```
Open the VoiceRAG folder in VS Code
```

---

### 3. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Set Up Ollama + Pull a Model

```bash
# Start Ollama (keep this running in background)
ollama serve

# In a new terminal, pull a model (choose one):
ollama pull mistral       # Recommended — fast and accurate (7B)
ollama pull llama3        # Meta's Llama 3 (8B)
ollama pull gemma         # Google's Gemma (7B)
ollama pull phi3          # Microsoft Phi-3 (lightweight)
```

---

### 6. Run the App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🗂️ Project Structure

```
VoiceRAG/
│
├── app.py                  # Streamlit UI (main entry point)
├── pdf_loader.py           # PDF text extraction & chunking
├── rag_pipeline.py         # Embeddings, FAISS, LLM integration
├── speech_to_text.py       # Whisper speech recognition
├── text_to_speech.py       # gTTS text-to-speech output
│
├── data/
│   └── uploaded_pdfs/      # PDFs are saved here
│
├── vector_db/              # FAISS index persisted here
│   ├── faiss.index
│   └── chunks.pkl
│
├── models/                 # (Optional) local model files
│
└── requirements.txt
```

---

## 🏗️ System Architecture

```
User (Voice/Text)
       │
       ▼
┌──────────────────┐
│  Streamlit UI    │  ← app.py
│  (app.py)        │
└──────┬───────────┘
       │
  ┌────▼────┐         ┌─────────────────┐
  │  Voice  │────────►│ Whisper (STT)   │
  │  Input  │         │ speech_to_text  │
  └─────────┘         └────────┬────────┘
                               │ Question Text
                               ▼
                    ┌──────────────────────┐
                    │  FAISS Retriever     │
                    │  rag_pipeline.py     │
                    │  (sentence-transformers)│
                    └──────────┬───────────┘
                               │ Top-K Chunks
                               ▼
                    ┌──────────────────────┐
                    │  Ollama LLM          │
                    │  (Mistral/Llama/etc) │
                    └──────────┬───────────┘
                               │ Answer
                    ┌──────────▼───────────┐
                    │   Display + gTTS     │
                    │   text_to_speech.py  │
                    └──────────────────────┘
```

---

## 🎮 How to Use

1. **Upload a PDF** — Click "Browse files" and select any PDF
2. **Process it** — Click "Process PDF" (builds the vector index)
3. **Ask a question**:
   - Click the 🎙️ microphone to speak your question, OR
   - Type your question in the text box
4. **Get the answer** — The LLM answers based on your document
5. **Listen** — The answer is spoken back using text-to-speech

---

## ⚙️ Configuration (Sidebar)

| Setting | Description | Default |
|---------|-------------|---------|
| LLM Model | Ollama model to use | mistral |
| Chunk Size | Characters per text chunk | 500 |
| Chunk Overlap | Overlap between chunks | 50 |
| Top-K Chunks | How many chunks to retrieve | 5 |

---

## 🧪 Test PDFs

Try with any of these:
- arXiv papers: https://arxiv.org (download PDF)
- Python docs: https://docs.python.org (PDF export)
- Any textbook chapter saved as PDF

---

## 🔧 Troubleshooting

**Ollama not connecting:**
```bash
# Make sure Ollama is running:
ollama serve
# Check available models:
ollama list
```

**Whisper errors:**
```bash
pip install openai-whisper
# Also install ffmpeg (see Prerequisites above)
```

**Voice recording not working:**
- Allow microphone permissions in your browser
- Use Chrome or Edge for best Web Audio API support

**PDF not extracting text:**
- The PDF must be text-based (not a scanned image)
- For scanned PDFs, you'd need OCR (Tesseract)

---

## 📚 Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| UI | Streamlit | Simple, fast Python web UI |
| PDF Parsing | pdfplumber + pypdf | Handles complex layouts |
| Chunking | LangChain TextSplitter | Smart recursive splitting |
| Embeddings | sentence-transformers | Local, no API key needed |
| Vector DB | FAISS | Fast similarity search |
| LLM | Ollama (Mistral/Llama) | Fully local, free |
| Speech-to-Text | OpenAI Whisper | Local ASR, very accurate |
| Text-to-Speech | gTTS | Simple, free |

---

## 📄 Resume Skills Demonstrated

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Vector databases (FAISS)
- ✅ LLM integration (Ollama)
- ✅ NLP & text embeddings
- ✅ Speech recognition (Whisper)
- ✅ AI application deployment (Streamlit)
- ✅ Python development
- ✅ AI system design & architecture

---

*Built for learning and portfolio use. All processing is 100% local.*
