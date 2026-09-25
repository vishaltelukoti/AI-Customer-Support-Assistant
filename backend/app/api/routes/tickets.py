from fastapi import APIRouter

from app.schemas.ticket import TicketCreate, TicketResponse
from app.services.ticket_service import receive_ticket

router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])


@router.post("", response_model=TicketResponse, summary="Process a support ticket")
def create_ticket(ticket: TicketCreate) -> TicketResponse:
    """Run the integrated Day 12 support-assistant workflow."""
    return receive_ticket(ticket)
