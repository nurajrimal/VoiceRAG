"""
app.py
VoiceRAG - Voice-Enabled PDF Question Answering System
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceRAG — PDF Q&A",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .status-ok   { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .status-warn { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .status-err  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .chunk-box {
        background: #fff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #495057;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Imports ──────────────────────────────────────────────────────────────────
from pdf_loader import load_and_process_pdf
from rag_pipeline import (
    build_vector_store, load_vector_store, vector_store_exists,
    answer_question, check_ollama_status, DEFAULT_LLM_MODEL
)
from speech_to_text import transcribe_audio_bytes, is_whisper_available
from text_to_speech import text_to_speech_bytes, is_tts_available

# ─── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "faiss_index": None,
    "chunks": None,
    "pdf_meta": None,
    "chat_history": [],
    "current_question": "",
    "pdf_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎙️ VoiceRAG</h1>
    <p style="margin:0; opacity:0.9;">Voice-Activated PDF Question Answering using RAG + Local LLM</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selection
    llm_model = st.selectbox(
        "🤖 LLM Model (Ollama)",
        ["mistral", "llama3", "llama3.1", "gemma", "gemma2", "phi3", "qwen2"],
        index=0,
        help="Make sure the model is pulled via: ollama pull <model>"
    )

    chunk_size = st.slider("📄 Chunk Size (chars)", 200, 1000, 500, 50,
                           help="Smaller = more precise retrieval. Larger = more context per chunk.")
    chunk_overlap = st.slider("🔗 Chunk Overlap (chars)", 0, 200, 50, 10)
    top_k = st.slider("🔍 Top-K Chunks to Retrieve", 1, 10, 5,
                      help="How many document chunks to pass to the LLM.")

    st.divider()
    st.header("📊 System Status")

    # Ollama status
    ollama_ok, ollama_msg = check_ollama_status(llm_model)
    status_class = "status-ok" if ollama_ok else "status-err"
    st.markdown(f'<div class="status-box {status_class}">{ollama_msg}</div>', unsafe_allow_html=True)

    # Whisper status
    whisper_ok = is_whisper_available()
    whisper_class = "status-ok" if whisper_ok else "status-warn"
    whisper_msg = "✅ Whisper ready" if whisper_ok else "⚠️ Whisper not installed\npip install openai-whisper"
    st.markdown(f'<div class="status-box {whisper_class}">{whisper_msg}</div>', unsafe_allow_html=True)

    # TTS status
    tts_ok = is_tts_available()
    tts_class = "status-ok" if tts_ok else "status-warn"
    tts_msg = "✅ gTTS ready" if tts_ok else "⚠️ gTTS not installed\npip install gTTS"
    st.markdown(f'<div class="status-box {tts_class}">{tts_msg}</div>', unsafe_allow_html=True)

    # PDF loaded status
    if st.session_state.pdf_loaded:
        meta = st.session_state.pdf_meta
        st.success(f"📑 **{meta['filename']}**\n{meta['pages']} pages | {len(st.session_state.chunks)} chunks")

    st.divider()
    st.caption("🔒 100% local processing — no data leaves your machine.")

# ─── Main Layout: Two Columns ─────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

# ══════════════════════════════════════
# LEFT COLUMN: PDF Upload
# ══════════════════════════════════════
with col_left:
    st.subheader("📂 Step 1 — Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — research paper, manual, report, textbook, etc."
    )

    if uploaded_file:
        save_dir = Path("data/uploaded_pdfs")
        save_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = save_dir / uploaded_file.name

        # Save uploaded file
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        process_btn = st.button("⚡ Process PDF", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("🔄 Extracting text and building vector store..."):
                try:
                    chunks, metadata = load_and_process_pdf(
                        str(pdf_path), chunk_size, chunk_overlap
                    )
                    index = build_vector_store(chunks)

                    st.session_state.faiss_index = index
                    st.session_state.chunks = chunks
                    st.session_state.pdf_meta = metadata
                    st.session_state.pdf_loaded = True
                    st.session_state.chat_history = []

                    st.success(f"✅ PDF processed! {len(chunks)} chunks indexed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")

    # Load existing vector store
    if not st.session_state.pdf_loaded and vector_store_exists():
        st.info("💾 A vector store from a previous session was found.")
        if st.button("📂 Load Previous Session", use_container_width=True):
            with st.spinner("Loading..."):
                index, chunks = load_vector_store()
                if index and chunks:
                    st.session_state.faiss_index = index
                    st.session_state.chunks = chunks
                    st.session_state.pdf_meta = {"filename": "Previous PDF", "pages": "?"}
                    st.session_state.pdf_loaded = True
                    st.rerun()

    if not uploaded_file and not st.session_state.pdf_loaded:
        st.markdown("""
        **Getting Started:**
        1. Upload a PDF above
        2. Click **Process PDF**
        3. Ask questions via voice or text
        4. Get AI-powered answers!

        **Supported PDFs:**
        - Research papers & arXiv articles
        - Textbooks & manuals
        - Reports & documentation
        - Any text-based PDF
        """)

# ══════════════════════════════════════
# RIGHT COLUMN: Q&A Interface
# ══════════════════════════════════════
with col_right:
    st.subheader("🎙️ Step 2 — Ask a Question")

    if not st.session_state.pdf_loaded:
        st.info("⬅️ Please upload and process a PDF first.")
    else:
        # ── Voice Input ──────────────────────────────────────────────────
        st.markdown("**🎤 Voice Input**")

        try:
            from streamlit_mic_recorder import mic_recorder
            audio = mic_recorder(
                start_prompt="🎙️ Click to Record",
                stop_prompt="⏹️ Stop Recording",
                key="mic"
            )
            if audio and audio.get("bytes"):
                with st.spinner("🔊 Transcribing audio..."):
                    if whisper_ok:
                        transcribed = transcribe_audio_bytes(
                            audio["bytes"],
                            sample_rate=audio.get("sample_rate", 16000)
                        )
                        if transcribed and not transcribed.startswith("[Error]"):
                            st.session_state.current_question = transcribed
                            st.success(f"🗣️ Heard: *\"{transcribed}\"*")
                        else:
                            st.warning(transcribed)
                    else:
                        st.warning("⚠️ Whisper not installed. Install it for voice input.")
        except ImportError:
            st.warning("⚠️ Install `streamlit-mic-recorder` for voice input:\n`pip install streamlit-mic-recorder`")

        # ── Text Input ───────────────────────────────────────────────────
        st.markdown("**✍️ Or Type Your Question**")
        text_question = st.text_area(
            "Type here",
            value=st.session_state.current_question,
            height=80,
            placeholder="e.g., What is the main contribution of this paper?",
            label_visibility="collapsed"
        )

        # ── Ask Button ───────────────────────────────────────────────────
        ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

        if ask_btn and text_question.strip():
            query = text_question.strip()
            st.session_state.current_question = ""

            with st.spinner("🤔 Thinking..."):
                answer, relevant_chunks = answer_question(
                    query,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    model=llm_model,
                    top_k=top_k
                )

            # Store in chat history
            st.session_state.chat_history.append({
                "question": query,
                "answer": answer,
                "chunks": relevant_chunks
            })

        # ── Chat History ─────────────────────────────────────────────────
        if st.session_state.chat_history:
            st.divider()
            st.subheader("💬 Conversation")

            # Show newest first
            for i, turn in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**🙋 You:** {turn['question']}")
                    st.markdown(f'<div class="answer-box">🤖 {turn["answer"]}</div>',
                                unsafe_allow_html=True)

                    # TTS playback
                    if tts_ok:
                        audio_bytes = text_to_speech_bytes(turn["answer"])
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

                    # Expandable source chunks
                    with st.expander(f"📑 View Source Chunks ({len(turn['chunks'])} retrieved)"):
                        for j, chunk in enumerate(turn["chunks"], 1):
                            st.markdown(f'<div class="chunk-box"><b>Chunk {j}:</b> {chunk}</div>',
                                        unsafe_allow_html=True)

                    if i < len(st.session_state.chat_history) - 1:
                        st.divider()

            # Clear history button
            if st.button("🗑️ Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#6c757d; font-size:0.85rem;">
    VoiceRAG · Built with Streamlit · Whisper · FAISS · LangChain · Ollama<br>
    All processing is local — your documents stay private.
</div>
""", unsafe_allow_html=True)