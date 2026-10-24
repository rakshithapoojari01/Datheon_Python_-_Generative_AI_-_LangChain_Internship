
from datetime import date

from pydantic import BaseModel, Field


class Invoice(BaseModel):
    invoice_number: str = Field(
        min_length=1,
        description="Invoice number"
    )

    vendor_name: str = Field(
        min_length=1,
        description="Name of the vendor"
    )

    invoice_date: date = Field(
        description="Invoice date in YYYY-MM-DD format"
    )

    total_amount: float = Field(
        ge=0,
        description="Total invoice amount"
    )

    currency: str = Field(
        min_length=3,
        max_length=3,
        description="Three-letter currency code, such as INR"
    )
