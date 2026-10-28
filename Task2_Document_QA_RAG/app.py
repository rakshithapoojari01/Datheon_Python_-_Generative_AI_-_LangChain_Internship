
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


# Load the API key from this task's own .env file
load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(
    page_title="Document QA Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document QA Assistant")
st.caption("Ask questions about your documents using RAG and vector embeddings.")


# Load the embedding model once and reuse it
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Extract text from each PDF page
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


# Split page text into overlapping chunks
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


# Build a FAISS similarity-search index
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


# Retrieve the most relevant chunks for a question
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


# Generate an answer using Gemini
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


# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Retrieved chunks", 1, 8, 4)
    st.info(
        "PDF text extraction, embeddings, and similarity search "
        "run locally. Gemini receives the question and retrieved text."
    )


# Upload a PDF
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file:
    if (
        "document_name" not in st.session_state
        or st.session_state.document_name != uploaded_file.name
    ):
        st.session_state.pop("chunks", None)
        st.session_state.pop("faiss_index", None)
        st.session_state.document_name = uploaded_file.name

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process document", type="primary"):
        try:
            with st.spinner("Extracting text from the PDF..."):
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
                    "Scanned PDFs may require OCR."
                )
            else:
                with st.spinner("Creating embeddings and search index..."):
                    chunks = split_into_chunks(pages)

                    if not chunks:
                        st.error("No usable text chunks were found.")
                    else:
                        model = load_embedding_model()
                        index = build_faiss_index(chunks, model)

                        st.session_state.chunks = chunks
                        st.session_state.faiss_index = index

                        st.success(
                            f"Processed {len(pages)} pages into "
                            f"{len(chunks)} searchable chunks."
                        )

        except Exception as error:
            st.error(f"Document processing failed: {error}")


# Ask questions about the processed document
if (
    "chunks" in st.session_state
    and "faiss_index" in st.session_state
):
    st.divider()
    st.subheader("💬 Ask your document")

    question = st.text_input(
        "Enter your question",
        placeholder="What are the main points in this document?",
    )

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                # Measure local retrieval time separately
                retrieval_start = time.perf_counter()

                with st.spinner("Searching the document..."):
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

                st.caption(
                    f"Document search time: "
                    f"{retrieval_elapsed:.2f} seconds"
                )

                # Measure Gemini generation time
                with st.spinner("Generating an answer with Gemini..."):
                    generation_start = time.perf_counter()

                    answer = generate_answer(
                        question,
                        retrieved,
                    )

                    generation_elapsed = (
                        time.perf_counter() - generation_start
                    )

                st.caption(
                    f"Gemini response time: "
                    f"{generation_elapsed:.2f} seconds"
                )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### 📖 Retrieved sources")

                for number, chunk in enumerate(retrieved, start=1):
                    with st.expander(
                        f"Source {number} — Page {chunk['page']} "
                        f"(similarity: {chunk['score']:.3f})"
                    ):
                        st.write(chunk["text"])

                
            except Exception as error:
                error_text = str(error)

                if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                    st.error(
                        "Gemini API quota exceeded. "
                        "Please wait for your quota to reset before trying again."
                    )
                elif "503" in error_text or "UNAVAILABLE" in error_text:
                    st.warning(
                        "Gemini is temporarily busy. Please try again later."
                    )
                elif "504" in error_text or "DEADLINE_EXCEEDED" in error_text:
                    st.warning(
                        "Gemini took too long to respond. Please try again later."
                    )
                else:
                    st.error(
                        f"Could not answer the question: {error_text}"
                    )         

else:
    st.info(
        "Upload a PDF and select Process document to get started."
    )
