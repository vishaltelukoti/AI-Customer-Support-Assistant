from fastapi import APIRouter

from app.schemas.ticket import TicketCreate, TicketResponse
from app.services.ticket_service import receive_ticket

router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])


@router.post("", response_model=TicketResponse, summary="Receive a ticket (Day 1 placeholder)")
def create_ticket(ticket: TicketCreate) -> TicketResponse:
    """Return an acknowledgement. No persistence or AI processing is performed."""
    return receive_ticket(ticket)
