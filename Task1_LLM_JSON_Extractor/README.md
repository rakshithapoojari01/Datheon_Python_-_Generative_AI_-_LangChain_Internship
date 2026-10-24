# Structured LLM Prompt Pipeline & JSON Extractor

## Overview
This project extracts structured invoice information from text using Google's Gemini API and validates the extracted data with Pydantic.

## Features
- Extracts invoice number, vendor name, invoice date, total amount, and currency.
- Uses Gemini to generate structured JSON output.
- Validates invoice fields using Pydantic.
- Rejects invalid dates and negative invoice amounts.

## Technologies Used
- Python
- Google Gemini API (`google-genai`)
- Pydantic
- python-dotenv

## Project Files
- `main.py` — Extracts invoice data using Gemini.
- `schemas.py` — Defines the invoice validation schema.
- `test_schema.py` — Tests valid invoice data.
- `test_invalid_invoice.py` — Tests invalid invoice data.
- `.gitignore` — Excludes secrets and temporary files.

## Setup

1. Install Python 3.11 or a compatible version.
2. Install the dependencies:

   ```bash
   pip install google-genai python-dotenv pydantic
   ```

3. Create a `.env` file in the project folder and add your own Gemini API key:

   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

4. Run the application:

   ```bash
   python main.py
   ```

## Security
Never commit your API key or `.env` file to GitHub.

## Status
Task 1 — implementation and basic validation tests completed.
