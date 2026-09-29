import type { MonitoringResponse } from '../types/ticket'

interface MonitoringPanelProps {
  monitoring: MonitoringResponse | null
  loading: boolean
  unavailable: boolean
  onRefresh: () => void
}

export function MonitoringPanel({ monitoring, loading, unavailable, onRefresh }: MonitoringPanelProps) {
  return (
    <section className="monitoring-panel" aria-labelledby="monitoring-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">POC evidence</p>
          <h2 id="monitoring-heading">Model and monitoring</h2>
        </div>
        <button className="secondary-button" type="button" onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>
      {monitoring ? (
        <div className="monitoring-grid">
          <dl className="monitoring-list">
            <div><dt>API requests</dt><dd>{monitoring.request_count}</dd></div>
            <div><dt>API errors</dt><dd>{monitoring.error_count}</dd></div>
            <div><dt>Average API latency</dt><dd>{monitoring.average_latency_ms.toFixed(1)} ms</dd></div>
            <div><dt>Model predictions</dt><dd>{monitoring.model_prediction_count}</dd></div>
            <div><dt>Retrieval requests</dt><dd>{monitoring.retrieval_request_count}</dd></div>
            <div><dt>Average retrieval latency</dt><dd>{monitoring.average_retrieval_latency_ms.toFixed(1)} ms</dd></div>
          </dl>
          <div className="system-info">
            <h3>Model / system information</h3>
            <dl>
              <div><dt>Classification</dt><dd>Day 3 optimized TF-IDF + Logistic Regression</dd></div>
              <div><dt>Retrieval embedding</dt><dd>all-MiniLM-L6-v2</dd></div>
              <div><dt>Retrieval Top-K</dt><dd>3</dd></div>
              <div><dt>Category Macro-F1</dt><dd>{(monitoring.model_quality.value * 100).toFixed(2)}%</dd></div>
              <div><dt>Retrieval Recall@3</dt><dd>{(monitoring.retrieval_quality.value * 100).toFixed(2)}%</dd></div>
            </dl>
          </div>
        </div>
      ) : (
        <p className="result-note">{unavailable ? 'Monitoring is unavailable while the backend is offline.' : 'Loading local monitoring data...'}</p>
      )}
      <p className="result-note">Runtime counters are in memory and reset when the backend restarts.</p>
    </section>
  )
}
