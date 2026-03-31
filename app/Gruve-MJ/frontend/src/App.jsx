import { useState } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || ''

function App() {
  const [file, setFile] = useState(null)
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictResult, setPredictResult] = useState(null)
  const [selectedRow, setSelectedRow] = useState(null)
  const [rowExplain, setRowExplain] = useState(null)
  const [rowExplainLoading, setRowExplainLoading] = useState(false)
  const [chatInput, setChatInput] = useState('')
  const [chatMessages, setChatMessages] = useState([])
  const [chatLoading, setChatLoading] = useState(false)
  const [vlmSummary, setVlmSummary] = useState('')
  const [vlmLoading, setVlmLoading] = useState(false)
  const [error, setError] = useState('')

  const healthHref = `${API_BASE}/health`

  const handlePredict = async (event) => {
    event.preventDefault()
    setError('')
    setPredictLoading(true)
    setPredictResult(null)
    setSelectedRow(null)
    setRowExplain(null)

    try {
      if (!file) {
        throw new Error('Please choose a CSV file first.')
      }
      const form = new FormData()
      form.append('file', file)
      form.append('include_input', 'true')
      form.append('explain_rows', '0')
      form.append('explain_top_n', '8')
      form.append('max_rows', '200')

      const { data } = await axios.post(`${API_BASE}/api/predict`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setPredictResult(data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Prediction failed.')
    } finally {
      setPredictLoading(false)
    }
  }

  const handleSelectRow = async (row) => {
    setSelectedRow(row)
    setRowExplainLoading(false)
    setRowExplain(null)
    setError('')

    if (!row.is_representative) {
      setRowExplain({
        predicted_label: row.predicted_label,
        confidence: row.confidence,
        local_shap: row.local_shap || [],
        llm_explanation:
          'LLM explanation is limited to representative rows. Select a representative row for full narrative.',
      })
      return
    }

    setRowExplainLoading(true)

    try {
      const payload = {
        row_id: row.row_id,
        row: row.input || {},
        predicted_label: row.predicted_label,
        top_n: 10,
        llm_model: 'gpt-4o-mini',
      }
      const { data } = await axios.post(`${API_BASE}/api/explain-row`, payload)
      setRowExplain(data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Row explanation failed.')
    } finally {
      setRowExplainLoading(false)
    }
  }

  const handleChat = async (event) => {
    event.preventDefault()
    if (!chatInput.trim()) {
      return
    }
    if (!selectedRow?.is_representative) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: 'Chat is available only for representative rows. Select a representative row first.',
        },
      ])
      return
    }
    const question = chatInput.trim()
    setChatInput('')
    setChatMessages((prev) => [...prev, { role: 'user', text: question }])
    setChatLoading(true)

    try {
      const context = {
        summary: predictResult?.summary || null,
        selected_row: selectedRow || null,
        selected_row_explanation: rowExplain || null,
      }
      const { data } = await axios.post(`${API_BASE}/api/chat`, {
        question,
        context,
        llm_model: 'gpt-4o-mini',
      })
      setChatMessages((prev) => [...prev, { role: 'assistant', text: data.answer }])
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Chat request failed.'
      setChatMessages((prev) => [...prev, { role: 'assistant', text: msg }])
    } finally {
      setChatLoading(false)
    }
  }

  const handleVlmSummary = async () => {
    setVlmLoading(true)
    setError('')
    try {
      const { data } = await axios.post(`${API_BASE}/api/vlm-summary`, {
        chart_paths: [
          'outputs/training/confusion_matrix_heatmap.png',
          'outputs/training/classwise_f1_bar.png',
          'outputs/training/classwise_recall_bar.png',
        ],
        prompt:
          'Summarize modelling implications from these evaluation charts in plain English for a hiring evaluator.',
        llm_model: 'gpt-4o-mini',
      })
      setVlmSummary(data.summary || '')
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'VLM summary failed.')
    } finally {
      setVlmLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">Gruve Submission App</p>
        <h1>Readmission Inference + Explainability Studio</h1>
        <p className="sub">
          Upload <code>unseen_data.csv</code>, inspect confidence and local SHAP drivers, and ask
          model-grounded questions in chat.
        </p>
        <a className="health-link" href={healthHref} target="_blank" rel="noreferrer">
          API Health Check
        </a>
      </header>

      <section className="card">
        <h2>1) CSV Upload & Batch Prediction</h2>
        <form className="upload-form" onSubmit={handlePredict}>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button type="submit" disabled={predictLoading}>
            {predictLoading ? 'Running...' : 'Run Inference'}
          </button>
        </form>
        {predictResult && (
          <div className="summary-grid">
            <div>
              <p className="k">Total rows</p>
              <p className="v">{predictResult.summary.rows_total}</p>
            </div>
            <div>
              <p className="k">Returned rows</p>
              <p className="v">{predictResult.summary.rows_returned}</p>
            </div>
            <div>
              <p className="k">Model path</p>
              <p className="v small">{predictResult.summary.model_path}</p>
            </div>
            <div>
              <p className="k">Macro summary</p>
              <p className="v small">
                {predictResult.summary.f1_macro
                  ? `F1(macro): ${predictResult.summary.f1_macro.toFixed(4)}`
                  : 'Label not provided'}
              </p>
            </div>
            <div>
              <p className="k">Representative rows</p>
              <p className="v">{predictResult.summary.representative_rows}</p>
            </div>
          </div>
        )}
      </section>

      {predictResult && (
        <section className="card">
          <h2>2) Prediction Table (click a row to explain)</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>row_id</th>
                  <th>prediction</th>
                  <th>confidence</th>
                  <th>p(&lt;30)</th>
                  <th>p(&gt;30)</th>
                  <th>p(NO)</th>
                </tr>
              </thead>
              <tbody>
                {predictResult.rows.map((row) => (
                  <tr
                    key={row.row_id}
                    onClick={() => handleSelectRow(row)}
                    className={selectedRow?.row_id === row.row_id ? 'active-row' : ''}
                  >
                    <td>{row.row_id}</td>
                    <td>
                      {row.predicted_label}
                      {row.is_representative ? <span className="rep-badge">REP</span> : null}
                    </td>
                    <td>{row.confidence.toFixed(4)}</td>
                    <td>{(row.probabilities['<30'] ?? 0).toFixed(4)}</td>
                    <td>{(row.probabilities['>30'] ?? 0).toFixed(4)}</td>
                    <td>{(row.probabilities.NO ?? 0).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {selectedRow && (
        <section className="card">
          <h2>3) Local Explanation (SHAP + LLM)</h2>
          {rowExplainLoading && <p>Generating explanation...</p>}
          {rowExplain && (
            <>
              <p className="sub">
                Prediction: <strong>{rowExplain.predicted_label}</strong> | Confidence:{' '}
                <strong>{rowExplain.confidence.toFixed(4)}</strong>
              </p>
              <div className="shap-grid">
                {rowExplain.local_shap.map((item) => (
                  <div key={item.feature} className="shap-item">
                    <p className="feature">{item.feature}</p>
                    <p className={item.shap_value >= 0 ? 'up' : 'down'}>
                      {item.shap_value.toFixed(4)} ({item.direction})
                    </p>
                  </div>
                ))}
              </div>
              <div className="markdown-box">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {rowExplain.llm_explanation}
                </ReactMarkdown>
              </div>
            </>
          )}
        </section>
      )}

      <section className="card">
        <h2>4) Chat with Model Output</h2>
        <div className="chat-log">
          {chatMessages.length === 0 && <p className="muted">Ask about predictions, confidence, or SHAP factors.</p>}
          {chatMessages.map((msg, idx) => (
            <div key={idx} className={`bubble ${msg.role}`}>
              <strong>{msg.role === 'user' ? 'You' : 'Assistant'}</strong>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
            </div>
          ))}
        </div>
        <form className="chat-form" onSubmit={handleChat}>
          <input
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Why was this row predicted as <30?"
          />
          <button type="submit" disabled={chatLoading}>
            {chatLoading ? 'Thinking...' : 'Send'}
          </button>
        </form>
        {!selectedRow?.is_representative ? (
          <p className="muted">Select a representative row (REP badge) to use chat.</p>
        ) : null}
      </section>

      <section className="card">
        <h2>5) VLM-style EDA Interpretation</h2>
        <p className="sub">
          Generates evaluator-facing interpretation text from configured EDA artifacts.
        </p>
        <button type="button" onClick={handleVlmSummary} disabled={vlmLoading}>
          {vlmLoading ? 'Summarizing...' : 'Generate EDA Interpretation'}
        </button>
        {vlmSummary && (
          <div className="markdown-box">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{vlmSummary}</ReactMarkdown>
          </div>
        )}
      </section>

      {error && (
        <section className="card error-card">
          <strong>Error:</strong> {error}
        </section>
      )}
    </div>
  )
}

export default App
