export interface TicketCreate {
  subject: string
  body: string
  top_k?: number
}

export interface TicketResponse {
  ticket: {
    ticket_id: string
    subject: string
    body: string
    ticket_text: string
  }
  classification: {
    category: string | null
    category_confidence: number | null
    priority: string | null
    priority_confidence: number | null
  }
  similar_tickets: Array<{
    ticket_id: string
    score: number
    category: string
    priority: string
    ticket_text: string
  }>
  workflow: {
    complexity: 'simple' | 'complex'
    type: 'simple_rag' | 'complex_multi_agent' | 'blocked' | 'error'
    trace: string[]
  }
  response: {
    answer: string
    sources: Array<{
      ticket_id: string
      similarity_score: number
      category: string
      priority: string
      excerpt?: string | null
    }>
    retrieval_status: string
  }
  explanation: {
    predicted_category: string | null
    confidence: number | null
    top_features: Array<{ feature: string; contribution: number }>
  }
  security: {
    allowed: boolean
    status: string
    reason: string
    category: string
    matched_rule: string
  }
  timings: {
    total_ms: number
    classification_ms: number
    retrieval_ms: number
    workflow_ms: number
  }
  status: 'completed' | 'blocked' | 'error'
}
