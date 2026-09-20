import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import App from './App.jsx'

// PathTracerView needs WebGL; swap it for a light stub so we can exercise the
// App's upload -> generate -> state logic under jsdom.
vi.mock('./PathTracerView.jsx', () => ({
  default: (props) => <div data-testid="ptv" data-has-glb={props.glb ? 'yes' : 'no'} />,
}))

function makeFile(name = 'plan.png', type = 'image/png') {
  return new File([new Uint8Array([137, 80, 78, 71])], name, { type })
}

describe('App', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })
  afterEach(() => cleanup())

  it('renders the panel and disabled generate button', () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
    const btn = screen.getByRole('button', { name: /Generate Path-Traced Dollhouse/i })
    expect(btn.disabled).toBe(true)
  })

  it('enables generate after picking a file', () => {
    render(<App />)
    const input = screen.getByAccept ? null : document.querySelector('input[type=file]')
    fireEvent.change(input, { target: { files: [makeFile()] } })
    const btn = screen.getByRole('button', { name: /Generate Path-Traced Dollhouse/i })
    expect(btn.disabled).toBe(false)
  })

  it('posts the floorplan and loads the returned GLB', async () => {
    const glbBytes = new Uint8Array([0x67, 0x6c, 0x54, 0x46, 1, 2, 3, 4]) // "glTF..."
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({
        'X-DolGen-Log': 'Extraction: 4 walls | Exported dollhouse GLB',
      }),
      arrayBuffer: async () => glbBytes.buffer,
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)
    const input = document.querySelector('input[type=file]')
    fireEvent.change(input, { target: { files: [makeFile()] } })
    fireEvent.click(screen.getByRole('button', { name: /Generate Path-Traced Dollhouse/i }))

    await waitFor(() => {
      expect(screen.getByTestId('ptv').getAttribute('data-has-glb')).toBe('yes')
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toBe('/generate-3d')
    expect(opts.method).toBe('POST')
    expect(opts.body.get('ceiling')).toBe('false')
    expect(opts.body.get('furniture_source')).toBeNull()
  })

  it('logs an error when the server rejects', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      text: async () => JSON.stringify({ detail: 'Kimi detected no walls in this floorplan.' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)
    fireEvent.change(document.querySelector('input[type=file]'), {
      target: { files: [makeFile()] },
    })
    fireEvent.click(screen.getByRole('button', { name: /Generate Path-Traced Dollhouse/i }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    // the GLB should NOT be set
    expect(screen.getByTestId('ptv').getAttribute('data-has-glb')).toBe('no')
  })
})
