# Datheon – Python & Generative AI Internship

## About the Internship

This repository contains the projects and practical tasks completed as part of the Datheon Python, Generative AI, and LangChain Internship.

The internship focuses on developing practical applications using Python, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), AI agents, and Streamlit.

## Tasks Completed

### Task 1 – LLM JSON Extractor
- Extracts structured information from unstructured text.
- Converts extracted information into JSON format.
- Includes schema-based validation.

### Task 2 – Document QA using RAG
- Implements document-based question answering.
- Uses Retrieval-Augmented Generation (RAG).
- Retrieves relevant document information to answer user questions.

### Task 3 – Multi-Tool Autonomous AI Agent
- Develops an AI agent capable of using multiple tools.
- Integrates tool-based functionality with an LLM.
- Demonstrates multi-step task execution.

### Task 4 – Streamlit AI Chatbot
- Builds an interactive AI chatbot using Streamlit.
- Integrates Google's Gemini model.
- Supports conversational interactions and chat history.
- Provides a customised user interface.

## Technologies Used

- Python
- LangChain
- Generative AI
- Large Language Models (LLMs)
- Google Gemini API
- Retrieval-Augmented Generation (RAG)
- Streamlit
- JSON
- Git and GitHub

## Repository Structure

```text
Datheon_Internship/
│
├── Task1_LLM_JSON_Extractor/
│
├── Task2_Document_QA_RAG/
│
├── Task3_Multi_Tool_AI_Agent/
│
├── Task4_Streamlit_AI_Chatbot/
│
├── .gitignore
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/rakshithapoojari01/Datheon_Python_-_Generative_AI_-_LangChain_Internship.git
```

2. Navigate to the repository:

```bash
cd Datheon_Python_-_Generative_AI_-_LangChain_Internship
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

4. Install the dependencies using the requirements file available inside the respective task folder.

## Running the Projects

Navigate to the required task folder and follow its README instructions.

For Task 4:

```bash
cd Task4_Streamlit_AI_Chatbot
pip install -r requirements.txt
python -m streamlit run app.py
```

## API Configuration

Some projects require API keys to access external AI services.

- Store API keys in a local `.env` file.
- Never upload API keys or secret credentials to GitHub.
- Refer to the individual task README files for configuration instructions.

## Author

**Rakshitha Poojari**

GitHub: [rakshithapoojari01](https://github.com/rakshithapoojari01)