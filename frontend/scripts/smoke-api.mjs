// Exercise the actual frontend Axios service against a running backend.
// This is an integration smoke check, not a browser/UI test.
import assert from 'node:assert/strict'
import { createServer } from 'vite'

const server = await createServer({
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, hmr: false },
})

try {
  const { submitTicket, getApiError } = await server.ssrLoadModule('/src/services/api.ts')
  if (process.argv.includes('--unavailable')) {
    await assert.rejects(submitTicket({ ticket_text: 'Sample unavailable API check.' }), (error) => {
      assert.match(getApiError(error), /Unable to reach the support API/)
      return true
    })
    console.log('PASS: frontend service reports an unavailable backend.')
  } else {
    const ticket = await submitTicket({ ticket_text: 'I was charged twice for the same order.' })
    assert.equal(ticket.status, 'received')
    assert.equal(ticket.ticket_text, 'I was charged twice for the same order.')
    assert.match(ticket.ticket_id, /^[0-9a-f-]{36}$/)
    assert.equal(ticket.category, null)
    assert.equal(ticket.priority, null)
    assert.equal(ticket.confidence, null)
    await assert.rejects(submitTicket({ ticket_text: '   ' }), (error) => {
      assert.equal(error.response?.status, 422)
      assert.match(getApiError(error), /between 1 and 10,000 characters/)
      return true
    })
    console.log('PASS: frontend Axios service receives tickets and handles API validation errors.')
  }
} finally {
  await server.close()
}
