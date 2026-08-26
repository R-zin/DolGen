# DolGen — 2D Floorplan → Path-Traced 3D Dollhouse

Upload a 2D floorplan image; a Modal backend uses **Gemini 2.5 Pro** to extract
walls / doors / windows / furniture as normalized 2D boxes, downloads matching
3D furniture assets from a Sketchfab-compatible API, assembles a textured
dollhouse with **Trimesh** (2.5 m walls, boolean-cut openings, white PBR walls,
light-wood floor), and streams back a `.glb`. The frontend renders it with a
custom Three.js viewer using **three-gpu-pathtracer** for progressive,
photorealistic path tracing.

## One-time setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs the modal CLI

# Secrets (keys stay on your machine — never committed here):
modal secret create gemini-secret GEMINI_API_KEY=<your-gemini-key>
modal secret create asset-api-secret ASSET_API_TOKEN=<your-asset-library-token>
# optional, point at your own GLB asset library instead of Sketchfab:
#   modal secret create asset-api-secret ASSET_API_TOKEN=... ASSET_BASE_URL=https://your.api/v3
```

## Run

```bash
modal serve app.py     # hot-reload playground URL
modal deploy app.py    # persistent web endpoint
```

Open the printed URL, drop in a floorplan PNG/JPG, and click **Generate**.

## Architecture

| Step | Where | What |
|------|-------|------|
| Extraction | `extract_elements()` | `gemini-2.5-pro`, `temperature=0.0`, strict JSON schema → `[{element_type, furniture_class, asset_search_query, box_2d}]` with boxes normalized to 0-1000 |
| Assets | `download_assets()` | Sketchfab v3 flow: `GET /search?q=…&downloadable=true` → `POST /models/{uid}/download` → `GET` the glb url. **Any** failure skips that furniture piece gracefully |
| Assembly | `build_shell()` / `assemble_glb_bytes()` | 1000 units → 20 m; walls 2.5 m via `trimesh.creation.box`; doors/windows boolean-subtracted (manifold engine); furniture uniformly scaled to its footprint and grounded at z=0; white-PBR walls, light-wood-PBR floor |
| Serve | `POST /generate-3d` | `StreamingResponse` with `model/gltf-binary`; per-run pipeline notes in `X-DolGen-Log` header |
| Render | `GET /` | ES-module import map: `three`, `GLTFLoader`, `OrbitControls`, `three-mesh-bvh`, `WebGLPathTracer` from unpkg; `ACESFilmicToneMapping`, `pathTracer.setScene(gltf.scene, camera)`, `renderSample()` each frame |

## Notes

- **Sketchfab reality check**: the public Sketchfab API rarely allows downloading
  arbitrary `.glb`s (downloads need per-model grants). The client is implemented
  against the documented v3 shape; when it 404s/denies, the app logs and skips
  that piece and still returns the full wall/floor scene. Set `ASSET_BASE_URL`
  to any GLB-capable asset library to get real furniture.
- The image's 20 MB upload cap and the 64 MB per-asset cap guard Modal memory.
- `GET /` works locally too for pure frontend iteration (only POST needs secrets).
