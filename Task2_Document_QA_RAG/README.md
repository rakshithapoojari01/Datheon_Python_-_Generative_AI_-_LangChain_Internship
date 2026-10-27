# Document QA Assistant using RAG

## Overview

The Document QA Assistant is a Python-based application that allows users to upload a PDF document and ask questions about its contents.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from the uploaded document and generate answers using the Gemini API.

## Features

- Upload PDF documents through a Streamlit interface.
- Extract text from PDF pages using PyPDF.
- Split document text into overlapping chunks.
- Generate text embeddings using Sentence Transformers.
- Retrieve relevant passages using FAISS similarity search.
- Generate answers using the Gemini API.
- Display retrieved source passages with page numbers.
- Measure document retrieval and answer-generation time separately.

## Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- python-dotenv
- NumPy

## Project Workflow

1. Upload a PDF document.
2. Extract text from the document.
3. Split the extracted text into smaller chunks.
4. Generate embeddings for the chunks.
5. Build a FAISS index for similarity search.
6. Enter a question about the document.
7. Retrieve the most relevant passages.
8. Send the question and retrieved context to Gemini.
9. Display the generated answer and source passages.

## Project Structure

```text
Task2_Document_QA_RAG/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

The `.env` file stores the API key locally and must not be uploaded to GitHub.

## Prerequisites

- Python 3.11 or another Python version supported by the installed dependencies.
- A Google Gemini API key.
- pip package installer.

## Installation and Setup

### 1. Open the project directory

```bash
cd Task2_Document_QA_RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Gemini API key

Create a `.env` file inside the project directory and add:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your own API key. Never commit or share the real key.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL where you can access the application in your browser.

## Usage

1. Open the application in your browser.
2. Upload a text-based PDF document.
3. Select **Process document**.
4. Enter a question about the uploaded document.
5. Select **Get answer**.
6. Review the generated answer and the retrieved source passages.

## How It Works

The application follows a Retrieval-Augmented Generation workflow.

- **Document processing:** Extracts text from PDF pages and divides it into overlapping chunks.
- **Embedding generation:** Converts document chunks into numerical vector representations.
- **Similarity search:** Uses FAISS to retrieve passages relevant to the user's question.
- **Answer generation:** Sends the retrieved context and question to Gemini to generate a grounded response.
- **Source display:** Shows the retrieved passages and their page numbers to help users verify the answer.

## Limitations

- Answer generation requires a working Gemini API connection and available API quota.
- Scanned PDFs without selectable text may require OCR.
- Answer quality depends on the information available in the document and the relevance of retrieved passages.
- Incorrect or incomplete source information may lead to incomplete answers.

## Future Improvements

- Support multiple PDF documents.
- Add conversation history.
- Improve document retrieval and relevance ranking.
- Add OCR support for scanned PDFs.
- Improve API error handling and response-time monitoring.

## Security

- Store API credentials in the local `.env` file.
- Keep `.env` and virtual environment folders out of version control.
- Do not publish API keys in source code, screenshots, or public repositories.

## Author

Developed as part of the Datheon Python and Generative AI internship.

