import { useState, type FormEvent } from 'react'
import { TicketResult } from '../components/TicketResult'
import { getApiError, submitTicket } from '../services/api'
import type { TicketResponse } from '../types/ticket'

export function SupportPage() {
  const [text, setText] = useState('')
  const [ticket, setTicket] = useState<TicketResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (loading) return
    setError('')
    setTicket(null)
    const ticketText = text.trim()
    if (!ticketText || Array.from(ticketText).length > 10000) {
      setError('Please enter a ticket between 1 and 10,000 characters.')
      return
    }
    setLoading(true)
    try {
      setTicket(await submitTicket({ ticket_text: ticketText }))
    } catch (cause) {
      setError(getApiError(cause))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="workspace">
      <header>
        <div className="brand"><span className="brand-mark" aria-hidden="true">S</span> SUPPORT WORKSPACE</div>
        <span className="phase">Day 1 · Foundation</span>
      </header>
      <div className="intro">
        <p className="eyebrow">A clearer starting point for support</p>
        <h1>AI Customer Support Assistant</h1>
        <p>Bring a customer issue into one simple workspace.</p>
      </div>
      <div className="scope-note"><strong>Foundation preview.</strong> Submit a ticket to test the workflow. AI analysis and ticket storage are not enabled yet.</div>
      <div className="workspace-grid">
        <section className="panel" aria-labelledby="new-ticket-heading">
          <h2 id="new-ticket-heading">New ticket</h2>
          <p className="section-description">Start with the customer’s message and any relevant context.</p>
          <form onSubmit={handleSubmit} noValidate aria-busy={loading}>
            <label htmlFor="ticket-text">Customer ticket</label>
            <textarea id="ticket-text" value={text} disabled={loading}
              onChange={(event) => { setText(event.target.value); setError(''); setTicket(null) }}
              placeholder="For example: I was charged twice for the same order."
              aria-describedby={`ticket-help${error ? ' ticket-error' : ''}`}
              aria-invalid={Boolean(error)} required rows={9} />
            <div className="input-help" id="ticket-help"><span>Include the details your support team needs.</span><span>{Array.from(text).length.toLocaleString()} / 10,000</span></div>
            {error && <p className="error" id="ticket-error" role="alert">{error}</p>}
            <button type="submit" disabled={loading}>{loading ? 'Submitting…' : 'Submit ticket'}<span aria-hidden="true"> →</span></button>
            <span className="loading-message" role="status">{loading ? 'Sending your ticket to the support API…' : ''}</span>
          </form>
        </section>
        <TicketResult ticket={ticket} />
      </div>
      <footer>Day 1 POC · Ticket intake and API foundation</footer>
    </main>
  )
}
