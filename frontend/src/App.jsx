import { useCallback, useRef, useState } from 'react'
import './App.css'
import PathTracerView from './PathTracerView.jsx'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [drag, setDrag] = useState(false)
  const [busy, setBusy] = useState(false)
  const [glb, setGlb] = useState(null)
  const [ceiling, setCeiling] = useState(false)
  const [spin, setSpin] = useState(false)
  const [quality, setQuality] = useState(1024)
  const [stats, setStats] = useState(null)
  const [logs, setLogs] = useState([])
  const viewRef = useRef(null)

  const log = useCallback((msg, cls = '') => {
    setLogs((prev) => [...prev.slice(-199), { id: prev.length + ':' + Date.now(), msg, cls }])
  }, [])

  const pick = useCallback(
    (f) => {
      if (!f) return
      setFile(f)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(URL.createObjectURL(f))
      log(`Selected ${f.name} (${(f.size / 1024).toFixed(0)} KB)`)
    },
    [previewUrl, log],
  )

  const generate = useCallback(async () => {
    if (!file || busy) return
    setBusy(true)
    log('Uploading floorplan — extracting structure with Kimi K3…')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('ceiling', ceiling ? 'true' : 'false')
      const res = await fetch('/generate-3d', { method: 'POST', body: form })
      if (!res.ok) {
        let detail = await res.text()
        try {
          detail = JSON.parse(detail).detail || detail
        } catch {
          /* keep raw text */
        }
        throw new Error(`server ${res.status}: ${detail}`)
      }
      const serverLog = res.headers.get('X-DolGen-Log')
      if (serverLog) {
        serverLog.split(' | ').forEach((l) => log(l, /failed/i.test(l) ? 'err' : ''))
      }

      const buf = await res.arrayBuffer()
      log(`Received ${(buf.byteLength / 1024).toFixed(0)} KB GLB — loading…`)
      setGlb(buf)
    } catch (err) {
      log(`Generation failed: ${err.message}`, 'err')
      if (/fetch|network|failed to fetch/i.test(err.message)) {
        log('Network error — is the backend running? (modal serve app.py)', 'err')
      }
    } finally {
      setBusy(false)
    }
  }, [file, busy, ceiling, log])

  const download = useCallback(() => {
    if (!glb) return
    const blob = new Blob([glb], { type: 'model/gltf-binary' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'dollhouse.glb'
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 2000)
    log('Saved dollhouse.glb', 'ok')
  }, [glb, log])

  const onLoaded = useCallback(
    ({ hasCeiling: hc, error }) => {
      if (error) {
        log(`Could not load GLB: ${error.message}`, 'err')
        return
      }
      log('Scene handed to path tracer — watch it converge.', 'ok')
      if (ceiling && !hc) log('No ceiling slab in this model (it was not generated).', 'err')
    },
    [ceiling, log],
  )

  const onStats = useCallback(
    (s) => {
      setStats(s)
      if (s.justConverged) log(`Path tracing converged at ${s.samples} samples.`, 'ok')
    },
    [log],
  )

  return (
    <div className="app">
      <aside className="panel">
        <h1>
          Dol<span>Gen</span> 🏠
        </h1>

        <div
          className={`drop${drag ? ' drag' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDrag(false)
            pick(e.dataTransfer.files[0])
          }}
        >
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => pick(e.target.files[0])}
          />
          {previewUrl ? (
            <img src={previewUrl} alt="floorplan preview" />
          ) : (
            <div className="hint">
              Drop a floorplan image here
              <br />
              or click to browse
            </div>
          )}
        </div>
        {file && <div className="filename">{file.name}</div>}

        <button className="go" disabled={!file || busy} onClick={generate}>
          Generate Path-Traced Dollhouse
        </button>

        <div className="row">
          <label>
            <input type="checkbox" checked={ceiling} onChange={(e) => setCeiling(e.target.checked)} />
            Ceiling / roof
          </label>
          <label>
            <input type="checkbox" checked={spin} onChange={(e) => setSpin(e.target.checked)} />
            Auto-rotate
          </label>
        </div>

        <div className="row">
          <label htmlFor="quality">Quality</label>
          <input
            id="quality"
            type="range"
            min="128"
            max="4096"
            step="128"
            value={quality}
            onChange={(e) => setQuality(parseInt(e.target.value, 10))}
          />
          <span className="val">{quality}</span>
        </div>

        <div className="row">
          <button className="dl" disabled={!glb} onClick={download}>
            ⬇ Download .glb
          </button>
          <button className="dl" disabled={!glb} onClick={() => viewRef.current?.resetView()}>
            ⟲ Reset view
          </button>
        </div>

        <div className="stats">
          {stats &&
            `samples: ${stats.samples} / ${quality}`}
          {stats?.converged && <span className="converged"> — converged</span>}
        </div>

        <div className="log">
          {logs.map((l) => (
            <div key={l.id} className={l.cls}>
              {l.msg}
            </div>
          ))}
        </div>
      </aside>

      <main className="view">
        <PathTracerView
          ref={viewRef}
          glb={glb}
          ceilingVisible={ceiling}
          autoRotate={spin}
          targetSamples={quality}
          onStats={onStats}
          onLoaded={onLoaded}
        />
        {!glb && (
          <div className="empty">
            <div className="big">📐 → 🏠</div>
            <div>
              Upload a floorplan to render your dollhouse.
              <br />
              Progressive path tracing will converge once the model loads.
            </div>
          </div>
        )}
        {glb && stats && (
          <div className="hud">
            {stats.samples} samples{stats.converged ? ' · converged' : ''}
          </div>
        )}
      </main>

      {busy && (
        <div className="busy">
          <div className="dots">
            <span>●</span>
            <span>●</span>
            <span>●</span>
          </div>
          <div className="busymsg">
            Analyzing structure with Kimi K3 — this can take ~10–30 s
          </div>
        </div>
      )}
    </div>
  )
}
