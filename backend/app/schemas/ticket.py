from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


TicketText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)
]
Category = Literal["Payment", "Refund", "Account", "Shipping", "Technical"]
Priority = Literal["Low", "Medium", "High"]


class TicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_text: TicketText


class TicketResponse(BaseModel):
    ticket_id: UUID
    ticket_text: TicketText
    category: Category | None = None
    priority: Priority | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: Literal["received"] = "received"
