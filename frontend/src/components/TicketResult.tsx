import type { TicketResponse } from '../types/ticket'

export function TicketResult({ ticket }: { ticket: TicketResponse | null }) {
  const classification = ticket?.classification
  return (
    <section className="panel result-panel" aria-labelledby="result-heading" aria-live="polite">
      <div className="panel-heading">
        <h2 id="result-heading">Ticket result</h2>
        <span className={`status ${ticket ? ticket.status : ''}`}>{ticket?.status ?? 'Awaiting ticket'}</span>
      </div>
      <dl className="metrics">
        <div><dt>Category</dt><dd>{classification?.category ?? 'Not available yet'}</dd></div>
        <div><dt>Priority</dt><dd>{classification?.priority ?? 'Not available yet'}</dd></div>
        <div><dt>Confidence</dt><dd>{classification?.category_confidence == null ? 'Not available yet' : `${Math.round(classification.category_confidence * 100)}%`}</dd></div>
      </dl>
      {ticket ? (
        <div className="result-stack">
          <div className="receipt">
            <p className="eyebrow">{ticket.security.status}</p>
            <p className="ticket-text"><strong>{ticket.ticket.subject}</strong><br />{ticket.ticket.body}</p>
            <p className="ticket-id">ID: {ticket.ticket.ticket_id}</p>
          </div>
          <section className="result-section">
            <h3>Workflow</h3>
            <p>{ticket.workflow.type === 'complex_multi_agent' ? 'Complex multi-agent' : ticket.workflow.type === 'simple_rag' ? 'Simple RAG' : ticket.workflow.type === 'blocked' ? 'Blocked' : 'Component fallback'} · {ticket.workflow.complexity}</p>
            {ticket.workflow.trace.length > 0 && <p className="trace">{ticket.workflow.trace.join(' → ')}</p>}
          </section>
          <section className="result-section">
            <h3>Resolution</h3>
            <p className="answer">{ticket.response.answer}</p>
            {ticket.response.sources.length > 0 && <ul className="compact-list">{ticket.response.sources.map((source) => (
              <li key={source.ticket_id}>{source.ticket_id} · {source.category} · {source.similarity_score.toFixed(3)}</li>
            ))}</ul>}
          </section>
          <section className="result-section">
            <h3>Similar tickets</h3>
            {ticket.similar_tickets.length ? <ul className="compact-list">{ticket.similar_tickets.map((item) => (
              <li key={item.ticket_id}>{item.ticket_id} · {item.category} · {item.score.toFixed(3)}</li>
            ))}</ul> : <p>No similar tickets returned.</p>}
          </section>
          <section className="result-section">
            <h3>Explanation</h3>
            {ticket.explanation.top_features.length ? <ul className="compact-list">{ticket.explanation.top_features.map((item) => (
              <li key={item.feature}>{item.feature} · {item.contribution.toFixed(3)}</li>
            ))}</ul> : <p>No explanation terms available.</p>}
          </section>
        </div>
      ) : <p className="empty-state">Submit a customer ticket to see its integrated result here.</p>}
      <p className="result-note">Responses are grounded in retrieved synthetic historical tickets and remain POC-only.</p>
    </section>
  )
}
