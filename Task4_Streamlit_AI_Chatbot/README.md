# Task 4: Interactive Full-Stack AI Chatbot with Streamlit

## Project Overview
An interactive AI chatbot built using Streamlit, LangChain, and Google's Gemini model. The application provides a browser-based chat interface with conversation history and streaming responses.

## Features
- Interactive chat interface
- Gemini-powered AI responses
- Conversation history using Streamlit session state
- Streaming response generation
- Clear conversation option
- Error handling for API failures

## Technologies Used
- Python
- Streamlit
- LangChain
- Google Gemini API
- python-dotenv

## Project Structure
```text
Task4_Streamlit_AI_Chatbot/
├── app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Install Python 3.11 or a compatible version.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file containing your Gemini API key, or configure the key securely using your environment.
4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Example
Enter a message in the chat input and receive an AI-generated response. The conversation history is retained during the current Streamlit session.

## Security
Never commit API keys or secret environment files to GitHub.

## Project Status
The chatbot has been launched locally, and basic conversation-history behaviour has been tested.
