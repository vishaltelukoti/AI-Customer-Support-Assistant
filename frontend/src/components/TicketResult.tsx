import type { TicketResponse } from '../types/ticket'

export function TicketResult({ ticket }: { ticket: TicketResponse | null }) {
  return (
    <section className="panel result-panel" aria-labelledby="result-heading" aria-live="polite">
      <div className="panel-heading">
        <h2 id="result-heading">Ticket result</h2>
        <span className={`status ${ticket ? 'received' : ''}`}>{ticket?.status ?? 'Awaiting ticket'}</span>
      </div>
      <dl className="metrics">
        <div><dt>Category</dt><dd>{ticket?.category ?? 'Not available yet'}</dd></div>
        <div><dt>Priority</dt><dd>{ticket?.priority ?? 'Not available yet'}</dd></div>
        <div><dt>Confidence</dt><dd>{ticket?.confidence == null ? 'Not available yet' : `${Math.round(ticket.confidence * 100)}%`}</dd></div>
      </dl>
      {ticket ? (
        <div className="receipt">
          <p className="eyebrow">Ticket received</p>
          <p className="ticket-text">{ticket.ticket_text}</p>
          <p className="ticket-id">ID: {ticket.ticket_id}</p>
        </div>
      ) : <p className="empty-state">Submit a customer ticket to see its acknowledgement here.</p>}
      <p className="result-note">AI classification is planned for a later phase. No prediction is generated in Day 1.</p>
    </section>
  )
}
