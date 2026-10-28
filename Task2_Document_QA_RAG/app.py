
import os
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(
    page_title="Document QA Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULTS = {
    "chunks": None,
    "faiss_index": None,
    "document_name": None,
    "page_count": 0,
    "chunk_count": 0,
    "last_search_time": None,
    "last_generation_time": None,
    "last_answer": None,
    "last_sources": [],
    "last_question": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# PREMIUM DASHBOARD STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --navy: #111827;
        --indigo: #4f46e5;
        --purple: #7c3aed;
        --muted: #64748b;
        --border: #e2e8f0;
    }

    .stApp {
        background: linear-gradient(
            135deg, #f8faff 0%, #eef2ff 100%
        );
        color: #172554;
    }

    .block-container {
    max-width: 1600px;
    width: 94%;
    padding-top: 2rem;
    padding-bottom: 3rem;
    }
    
    h1 {
        color: #172554 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #243b6b !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #10172a 0%, #211b4b 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    .hero-card {
        background: linear-gradient(
            120deg, #171e3c 0%, #3730a3 60%, #6d28d9 100%
        );
        padding: 30px;
        border-radius: 22px;
        color: white;
        margin: 4px 0 24px 0;
        box-shadow: 0 12px 35px rgba(49, 46, 129, 0.18);
    }

    .hero-eyebrow {
        color: #c7d2fe;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: white !important;
        font-size: 32px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 10px;
    }

    .hero-description {
        color: #e0e7ff;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-label {
        color: #6366f1;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 18px;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #e0e7ff;
        border-radius: 16px;
        padding: 17px 18px;
        box-shadow: 0 5px 18px rgba(30, 41, 59, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #312e81;
        font-weight: 800;
    }

    .stButton > button {
        background: linear-gradient(
            135deg, #4f46e5, #7c3aed
        );
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 43px;
        padding: 0.6rem 1.1rem;
        font-weight: 700;
        box-shadow: 0 5px 14px rgba(99, 102, 241, 0.18);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        color: white;
        border: none;
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.28);
    }

    [data-testid="stTextInput"] input {
        border: 1px solid #dbe3f3;
        border-radius: 11px;
        background: white;
        color: #172554;
        min-height: 45px;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #818cf8;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.12);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.85);
        border: 1px dashed #a5b4fc;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #dbe3f3;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stCaptionContainer"] {
        color: #64748b;
    }

    hr {
        border-color: #dbe3f3;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .hero-card {
            padding: 22px;
        }

        .hero-title {
            font-size: 25px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">
            AI DOCUMENT INTELLIGENCE
        </div>
        <div class="hero-title">
            Document QA Assistant
        </div>
        <p class="hero-description">
            Turn lengthy PDFs into useful answers.
            Ask questions, explore relevant passages,
            and trace answers back to their source pages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------

def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip(),
            })

    return pages


def split_into_chunks(pages, chunk_size=180, overlap=30):
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                })

            if end == len(words):
                break

            start = end - overlap

    return chunks


def build_faiss_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_chunks(question, chunks, index, model, top_k=4):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")
    k = min(top_k, len(chunks))

    scores, indices = index.search(query_embedding, k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                **chunks[int(idx)],
                "score": float(score),
            })

    return results


# --------------------------------------------------
# GEMINI ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to the Task2_Document_QA_RAG/.env file."
        )

    if not retrieved_chunks:
        return "No relevant passages were found in the document."

    context = "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=60000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the supplied document context.
Be clear, accurate, and sufficiently detailed.
Do not invent facts or make unsupported claims.
If the answer is not present in the context, clearly say that
you could not find the answer in the uploaded document.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    print("Sending request to Gemini...", flush=True)
    start_time = time.perf_counter()

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            ),
        )

        elapsed = time.perf_counter() - start_time
        print(
            f"Gemini request completed in {elapsed:.2f} seconds.",
            flush=True,
        )

        return response.text or "No answer was returned."

    except Exception as error:
        elapsed = time.perf_counter() - start_time
        print(
            f"Gemini request failed after {elapsed:.2f} seconds: {error}",
            flush=True,
        )
        raise


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.caption("DOCUMENT QA SETTINGS")

    top_k = st.slider(
        "Retrieved source chunks",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of document passages to retrieve for each question.",
    )

    st.divider()

    st.markdown("### 🔎 How it works")
    st.markdown(
        """
        1. **Upload** a PDF.
        2. **Process** its text.
        3. **Ask** a question.
        4. **Explore** the source passages.
        """
    )

    st.divider()

    st.markdown("### 🛡️ Privacy")
    st.caption(
        "PDF extraction, embeddings, and similarity search run locally. "
        "The question and retrieved passages are sent to Gemini "
        "when you request an answer."
    )


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-label">01 / DOCUMENT WORKSPACE</div>',
    unsafe_allow_html=True,
)

st.subheader("Upload your document")
st.markdown(
    '<p class="section-description">'
    'Upload a text-based PDF to build a searchable knowledge base.'
    '</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Select a PDF containing text that can be extracted.",
)

if uploaded_file:
    if st.session_state.document_name != uploaded_file.name:
        st.session_state.chunks = None
        st.session_state.faiss_index = None
        st.session_state.document_name = uploaded_file.name
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        st.session_state.last_search_time = None
        st.session_state.last_generation_time = None
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.session_state.last_question = None

    file_col, action_col = st.columns([4, 1])

    with file_col:
        st.info(
            f"📄 **{uploaded_file.name}** · "
            f"{uploaded_file.size / (1024 * 1024):.2f} MB"
        )

    with action_col:
        process_clicked = st.button(
            "⚡ Process PDF",
            type="primary",
            use_container_width=True,
        )

    if process_clicked:
        try:
            with st.spinner("Extracting text from your PDF..."):
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(uploaded_file.getvalue())
                    temporary_path = temporary_file.name

                try:
                    pages = extract_pdf_text(temporary_path)
                finally:
                    Path(temporary_path).unlink(missing_ok=True)

            if not pages:
                st.error(
                    "No selectable text was found. "
                    "This PDF may be scanned and require OCR."
                )
            else:
                chunks = split_into_chunks(pages)

                if not chunks:
                    st.error("No usable text chunks were found.")
                else:
                    with st.spinner(
                        "Generating embeddings and building the search index..."
                    ):
                        model = load_embedding_model()
                        index = build_faiss_index(chunks, model)

                    st.session_state.chunks = chunks
                    st.session_state.faiss_index = index
                    st.session_state.page_count = len(pages)
                    st.session_state.chunk_count = len(chunks)

                    st.session_state.last_search_time = None
                    st.session_state.last_generation_time = None
                    st.session_state.last_answer = None
                    st.session_state.last_sources = []
                    st.session_state.last_question = None

                    st.success(
                        f"Document processed successfully: "
                        f"{len(pages)} text pages and "
                        f"{len(chunks)} searchable chunks."
                    )

        except Exception as error:
            st.error(f"Document processing failed: {error}")


# --------------------------------------------------
# LIVE METRICS
# --------------------------------------------------

st.markdown("---")
st.markdown(
    '<div class="section-label">02 / KNOWLEDGE BASE</div>',
    unsafe_allow_html=True,
)
st.subheader("Document overview")

metric1, metric2, metric3 = st.columns(3)

is_processed = (
    st.session_state.chunks is not None
    and st.session_state.faiss_index is not None
)

with metric1:
    st.metric(
        "Document status",
        "Ready" if is_processed else "Not processed",
    )

with metric2:
    st.metric(
        "Indexed chunks",
        st.session_state.chunk_count if is_processed else 0,
    )

with metric3:
    search_time = st.session_state.last_search_time
    st.metric(
        "Last search",
        f"{search_time:.2f} s" if search_time is not None else "—",
    )

if is_processed:
    st.caption(
        f"Active document: {st.session_state.document_name} · "
        f"{st.session_state.page_count} pages containing text"
    )
else:
    st.caption(
        "Upload and process a PDF to populate your document statistics."
    )


# --------------------------------------------------
# QUESTION AND ANSWER WORKSPACE
# --------------------------------------------------

st.markdown("---")
st.markdown(
    '<div class="section-label">03 / ASK YOUR DOCUMENT</div>',
    unsafe_allow_html=True,
)
st.subheader("What would you like to know?")
st.markdown(
    '<p class="section-description">'
    'Ask a question in natural language. Answers are generated '
    'from relevant passages retrieved from your document.'
    '</p>',
    unsafe_allow_html=True,
)

if is_processed:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the main findings in this document?",
        key="question_input",
    )

    if st.button(
        "✨ Generate answer",
        type="primary",
        use_container_width=False,
    ):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Searching your document..."):
                    retrieval_start = time.perf_counter()
                    model = load_embedding_model()

                    retrieved = retrieve_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.faiss_index,
                        model,
                        top_k=top_k,
                    )

                    retrieval_elapsed = (
                        time.perf_counter() - retrieval_start
                    )

                st.session_state.last_search_time = retrieval_elapsed
                st.session_state.last_question = question
                st.session_state.last_sources = retrieved
                st.session_state.last_generation_time = None
                st.session_state.last_answer = None

                if not retrieved:
                    st.warning(
                        "No relevant passages were found in this document."
                    )
                else:
                    with st.spinner(
                        "Generating an answer with Gemini..."
                    ):
                        generation_start = time.perf_counter()

                        answer = generate_answer(
                            question,
                            retrieved,
                        )

                        generation_elapsed = (
                            time.perf_counter() - generation_start
                        )

                    st.session_state.last_answer = answer
                    st.session_state.last_generation_time = (
                        generation_elapsed
                    )

            except Exception as error:
                error_text = str(error)
                error_upper = error_text.upper()

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_upper
                    or "QUOTA" in error_upper
                ):
                    st.error(
                        "🚦 **Gemini API quota exceeded**\n\n"
                        "The current model's API quota is unavailable. "
                        "Wait for the quota to reset before trying again. "
                        "Your PDF index is still available."
                    )

                elif (
                    "503" in error_text
                    or "UNAVAILABLE" in error_upper
                ):
                    st.warning(
                        "⏳ **Gemini is temporarily unavailable.** "
                        "Please try again later. Your document index "
                        "is still available."
                    )

                elif (
                    "504" in error_text
                    or "DEADLINE_EXCEEDED" in error_upper
                    or "TIMEOUT" in error_upper
                ):
                    st.warning(
                        "⌛ **Gemini took too long to respond.** "
                        "Please try again later. Your document index "
                        "is still available."
                    )

                elif "GEMINI_API_KEY" in error_text:
                    st.error(
                        "Gemini API key is missing. Check the local "
                        "Task2_Document_QA_RAG/.env file."
                    )

                else:
                    st.error(
                        f"Could not generate an answer: {error_text}"
                    )

    # Show the last successful answer and its sources.
    if st.session_state.last_answer:
        st.markdown("### 💡 Answer")

        answer_col, time_col = st.columns([4, 1])

        with answer_col:
            st.markdown(st.session_state.last_answer)

        with time_col:
            generation_time = st.session_state.last_generation_time
            st.metric(
                "Generation time",
                f"{generation_time:.2f} s"
                if generation_time is not None
                else "—",
            )

    if st.session_state.last_sources:
        st.markdown("### 📚 Retrieved source passages")
        st.caption(
            "Review these passages to understand which parts of "
            "the document were retrieved for your question."
        )

        for number, chunk in enumerate(
            st.session_state.last_sources,
            start=1,
        ):
            with st.expander(
                f"Source {number} · Page {chunk['page']} · "
                f"Similarity {chunk['score']:.3f}"
            ):
                st.write(chunk["text"])

else:
    st.info(
        "📄 Upload a PDF and select **Process PDF** before asking questions."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style="
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 12px 0;">
        Document QA Assistant · Built with Streamlit, FAISS,
        Sentence Transformers and Gemini
    </div>
    """,
    unsafe_allow_html=True,
)
