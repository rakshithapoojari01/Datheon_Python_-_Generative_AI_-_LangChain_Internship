
from pydantic import ValidationError
from schemas import Invoice

try:
    invoice = Invoice(
        invoice_number="INV-003",
        vendor_name="ABC Technologies",
        invoice_date="not-a-date",
        total_amount=-100,
        currency="INR",
    )
    print("Invoice accepted:")
    print(invoice.model_dump_json(indent=2))

except ValidationError as error:
    print("Invalid invoice rejected!")
    print(error)
