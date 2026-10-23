
import json
import os

from dotenv import load_dotenv
from google import genai

from schemas import Invoice

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)


def extract_invoice(invoice_text: str) -> Invoice:
    prompt = f"""
Extract invoice information from the text below.

Return only a JSON object with these fields:
- invoice_number
- vendor_name
- invoice_date (YYYY-MM-DD)
- total_amount (number)
- currency (three-letter code, such as INR)

Do not invent missing information. If a required field
is missing or unclear, report that instead of guessing.

Invoice text:
{invoice_text}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        },
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    data = json.loads(response.text)
    return Invoice.model_validate(data)


if __name__ == "__main__":
    sample_text = """
    TAX INVOICE
    Invoice Number: INV-104
    Vendor: ABC Technologies
    Invoice Date: 2026-10-20
    Total Amount: INR 3500.75
    """

    try:
        invoice = extract_invoice(sample_text)

        print("Extracted and validated invoice:")
        print(invoice.model_dump_json(indent=2))

    except Exception as error:
        print(f"Invoice extraction failed: {error}")
