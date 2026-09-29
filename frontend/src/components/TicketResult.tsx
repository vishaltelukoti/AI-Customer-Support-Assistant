import type { TicketResponse } from '../types/ticket'

export function TicketResult({ ticket }: { ticket: TicketResponse | null }) {
  const classification = ticket?.classification
  const isAbstaining = ticket?.response.retrieval_status.includes('insufficient')
    || ticket?.response.retrieval_status.includes('failed')
  const workflowLabel = ticket?.workflow.type === 'complex_multi_agent'
    ? 'Complex multi-agent workflow'
    : ticket?.workflow.type === 'simple_rag'
      ? 'Simple retrieval to resolution path'
      : ticket?.workflow.type === 'blocked'
        ? 'Blocked before workflow execution'
        : 'Component fallback'

  return (
    <section className="panel result-panel" aria-labelledby="result-heading" aria-live="polite">
      <div className="panel-heading">
        <h2 id="result-heading">Ticket result</h2>
        <span className={`status ${ticket ? ticket.status : ''}`}>{ticket?.status ?? 'Awaiting ticket'}</span>
      </div>
      <dl className="metrics">
        <div><dt>Category</dt><dd>{classification?.category ?? 'Not available yet'}</dd></div>
        <div><dt>Priority</dt><dd>{classification?.priority ?? 'Not available yet'}</dd></div>
        <div><dt>Category confidence</dt><dd>{classification?.category_confidence == null ? 'Not available yet' : `${Math.round(classification.category_confidence * 100)}%`}</dd></div>
        <div><dt>Priority confidence</dt><dd>{classification?.priority_confidence == null ? 'Not available yet' : `${Math.round(classification.priority_confidence * 100)}%`}</dd></div>
      </dl>
      {ticket ? (
        <div className="result-stack">
          <div className="receipt">
            <p className={`security-status ${ticket.security.allowed ? 'security-safe' : 'security-blocked'}`}>
              {ticket.security.allowed ? 'Security passed' : 'Security blocked'}: {ticket.security.status}
            </p>
            <p className="ticket-text"><strong>{ticket.ticket.subject}</strong><br />{ticket.ticket.body}</p>
            <p className="ticket-id">ID: {ticket.ticket.ticket_id}</p>
          </div>
          <section className="result-section">
            <h3>Workflow</h3>
            <p>{workflowLabel} - Complexity: {ticket.workflow.complexity}</p>
            {ticket.workflow.trace.length > 0 && (
              <ol className="agent-trace">{ticket.workflow.trace.map((step) => <li key={step}>{step}</li>)}</ol>
            )}
          </section>
          {ticket.workflow.investigation_result && (
            <section className="result-section">
              <h3>Investigation</h3>
              <p>{ticket.workflow.investigation_result}</p>
            </section>
          )}
          <section className="result-section">
            <h3>Security</h3>
            <p>{ticket.security.reason}</p>
            <p className="result-note">Category: {ticket.security.category}</p>
          </section>
          <section className="result-section">
            <h3>Resolution</h3>
            {isAbstaining && <p className="abstention">Insufficient retrieved evidence: the assistant abstained rather than fabricating a resolution.</p>}
            <p className="answer">{ticket.response.answer}</p>
            <h4>Sources</h4>
            {ticket.response.sources.length > 0 ? (
              <ul className="compact-list">{ticket.response.sources.map((source) => (
                <li key={source.ticket_id}>
                  {source.title ?? source.ticket_id} <small className="source-tag">{source.source_type === 'knowledge_base' ? 'Knowledge base' : 'Historical ticket'}</small>
                  {' - '}{source.category}{' - '}{source.similarity_score.toFixed(3)}
                </li>
              ))}</ul>
            ) : <p className="result-note">No supporting sources were returned.</p>}
          </section>
          <section className="result-section">
            <h3>Similar historical tickets</h3>
            {ticket.similar_tickets.length ? (
              <ul className="compact-list">{ticket.similar_tickets.slice(0, 3).map((item) => (
                <li key={item.ticket_id}><strong>{item.ticket_id}</strong>{' - '}{item.category}{' - '}{item.priority}{' - similarity '}{item.score.toFixed(3)}</li>
              ))}</ul>
            ) : <p>No similar tickets returned.</p>}
          </section>
          <section className="result-section">
            <h3>Explanation</h3>
            {ticket.explanation.top_features.length ? (
              <ul className="compact-list">{ticket.explanation.top_features.map((item) => (
                <li key={item.feature}>{item.feature}{' - contribution '}{item.contribution.toFixed(3)}</li>
              ))}</ul>
            ) : <p>No explanation terms available.</p>}
          </section>
        </div>
      ) : <p className="empty-state">Submit a customer ticket to see its integrated result here.</p>}
      <p className="result-note">Responses are grounded in retrieved synthetic historical tickets and fictional knowledge-base content; they remain POC-only.</p>
    </section>
  )
}
