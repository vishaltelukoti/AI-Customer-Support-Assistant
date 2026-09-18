from uuid import uuid4

from app.schemas.ticket import TicketCreate, TicketResponse


def receive_ticket(ticket: TicketCreate) -> TicketResponse:
    """Acknowledge a ticket without storing it or producing AI predictions."""
    return TicketResponse(ticket_id=uuid4(), ticket_text=ticket.ticket_text)
