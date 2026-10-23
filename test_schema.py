
from schemas import Invoice

invoice = Invoice(
    invoice_number="INV-001",
    vendor_name="ABC Technologies",
    invoice_date="2026-10-20",
    total_amount=2500.50,
    currency="INR",
)

print(invoice.model_dump_json(indent=2))
