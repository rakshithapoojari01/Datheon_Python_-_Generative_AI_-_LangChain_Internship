# Task 3: Multi-Tool Autonomous AI Agent

## Project Overview
This project implements a multi-tool AI agent using LangChain and Google's Gemini model. The agent can select appropriate tools to answer user queries.

## Features
- **Web Search:** Retrieves information from the web using DuckDuckGo.
- **Calculator:** Evaluates basic mathematical expressions.
- **Local File Reader:** Reads text files within the project directory.
- **AI-Powered Responses:** Uses Gemini to interpret queries and formulate responses.

## Technologies Used
- Python
- LangChain
- Google Gemini API
- DuckDuckGo Search (`ddgs`)
- python-dotenv

## Project Structure
```text
Task3_Multi_Tool_AI_Agent/
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Setup Instructions

1. Install Python 3.11 or a compatible version.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in this folder and add your Gemini API key:

   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. Run the agent:

   ```bash
   python main.py
   ```

5. Enter a question at the prompt. Type `exit` to quit.

## Example Queries
- `Calculate (125 * 8) + 450`
- `Search the web for recent developments in artificial intelligence`
- `Read the contents of README.md`

## Security
Keep API keys private. Do not commit `.env` or other files containing secrets to GitHub.

## Project Status
The calculator has been tested successfully. Web search and local file reading are being validated.
