export interface TicketCreate {
  ticket_text: string
}

export interface TicketResponse {
  ticket_id: string
  ticket_text: string
  category: 'Payment' | 'Refund' | 'Account' | 'Shipping' | 'Technical' | null
  priority: 'Low' | 'Medium' | 'High' | null
  confidence: number | null
  status: 'received'
}
