import axios from 'axios'
import type { TicketCreate, TicketResponse } from '../types/ticket'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 15000,
})

export async function submitTicket(ticket: TicketCreate): Promise<TicketResponse> {
  const response = await api.post<TicketResponse>('/api/v1/tickets', ticket)
  return response.data
}

export function getApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.code === 'ECONNABORTED') return 'The request timed out. Please try again.'
    if (!error.response) return 'Unable to reach the support API. Check that the backend is running and try again.'
    if (error.response.status === 422) return 'Please enter a subject and ticket body between 1 and 10,000 characters.'
    return `The API could not receive this ticket (HTTP ${error.response.status}). Please try again.`
  }
  return 'Something went wrong. Please try again.'
}
