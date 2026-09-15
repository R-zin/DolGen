"""
DolGen — turn a 2D floorplan image into a path-traced 3D dollhouse.

Pipeline (per POST /generate-3d request):
  1. Gemini 2.5 Pro (temperature=0.0, strict JSON schema) extracts walls / doors /
     windows / furniture as 2D boxes normalized to a 0-1000 image coordinate system.
  2. Furniture elements are resolved to .glb models. In "auto" mode each piece
     first tries a Sketchfab-compatible asset API, then falls back to an
     AI-generated model: the furniture's footprint is cropped from the floorplan,
     Gemini 3 Pro Image ("nano banana pro") renders it as a clean top-down
     product image, and that image is texture-mapped onto a procedural mesh and
     exported as a .glb. Any failure falls back to an untextured procedural mesh.
  3. Trimesh assembles the dollhouse: 2.5 m walls, boolean-cut door/window
     openings, PBR materials (white walls, light-wood floor), furniture scaled
     and placed inside its 2D footprint.
  4. The merged scene is exported as binary GLTF (.glb) and streamed back with
     media type 'model/gltf-binary'.

GET / serves a legacy Three.js viewer (zero-build fallback). The primary
frontend is the React app in frontend/ (Vite) which renders the GLB with
three-gpu-pathtracer (progressive path tracing + ACES filmic tone mapping).

Modal setup (secrets are intentionally NOT baked into this file):
  modal secret create gemini-secret GEMINI_API_KEY=...
  modal secret create asset-api-secret ASSET_API_TOKEN=...   # optional ASSET_BASE_URL=...
  modal serve app.py      # hot-reload playground URL
  modal deploy app.py     # persistent deployment
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
from typing import Any, Literal, Optional

import httpx
import modal
import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator

try:  # precise error type when the installed google-genai exposes it
    from google.genai.errors import ClientError as GenAIClientError
except Exception:  # pragma: no cover - older SDKs
    GenAIClientError = None  # type: ignore[assignment]

logger = logging.getLogger("dolgen")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Modal environment
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # runtime libs trimesh's vision stack needs
    .pip_install(
        "fastapi",
        "python-multipart",
        "google-genai",
        "openai",
        # trimesh[all] covers the glTF stack; manifold3d lives in trimesh's *easy*
        # extra, not [all], so it must be pinned explicitly for boolean operations.
        "trimesh[all]",
        "manifold3d>=2.3.0",
        "scipy",
        "networkx",
        "httpx",
        "pillow",  # crop floorplan + build texture images for AI furniture
        # Open-weights furniture renderer (FLUX.1 Kontext [dev]) — torch/cuda
        # come preinstalled on Modal's debian_slim GPU base, diffusers pulls
        # the rest. Imported lazily inside the GPU class so the CPU web
        # container never pays the import cost.
        "torch",
        "diffusers>=0.35.0",
        "transformers",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
)

app = modal.App(name="dolgen", image=image)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 1000 normalized image units  ==  20 physical meters
VIEW_UNITS_METERS = 20.0
SCALE = VIEW_UNITS_METERS / 1000.0

WALL_HEIGHT = 2.5          # meters
WALL_THICKNESS = 0.15      # meters
FLOOR_THICKNESS = 0.10     # meters (floor slab extends downward from z=0)

# Floorplan-extraction LLM: Kimi K3 (vision) served as an OpenAI-compatible
# Modal proxy endpoint. The proxy authenticates with the workspace's
# MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET joined by a dot.
KIMI_BASE_URL = os.environ.get(
    "KIMI_BASE_URL", "https://content-do--ep-kimi-k3-server.us-west.modal.direct/v1"
)
KIMI_MODEL = os.environ.get("KIMI_MODEL", "moonshotai/Kimi-K3")


def _kimi_api_key() -> str:
    """Modal proxy endpoints take '<token_id>.<token_secret>' as the API key."""
    token_id = os.environ.get("MODAL_PROXY_TOKEN_ID", "wk-FdegwHxb4Ut1MxVa3YZsyr")
    token_secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET", "ws-5FcFsksBUoYtOgOxWPXCBy")
    return f"{token_id}.{token_secret}"
# Image model ("nano banana pro" / Gemini 3 Pro Image) used to render per-furniture
# top-down views for AI-generated GLB textures. Overridable because Google renames
# preview model IDs as they go stable.
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")

# Which backend renders AI furniture top-down views:
#   "gemini"       (default) — Gemini image model, per-call pricing, no GPU
#   "openweights"            — FLUX.1 Kontext [dev] on a Modal GPU container
# Containers scale to zero when idle (scaledown_window=2s, the smallest Modal
# accepts): every cold start pays the model-load time, nothing is paid idle.
AI_IMAGE_BACKEND = os.environ.get("AI_IMAGE_BACKEND", "gemini").strip().lower()
OPENWEIGHTS_MODEL = os.environ.get("OPENWEIGHTS_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")
# Volume caching the HF weights so cold starts after the first are seconds,
# not a 30+ GB download. Created automatically on first deploy.
WEIGHTS_VOLUME_NAME = os.environ.get("WEIGHTS_VOLUME_NAME", "dolgen-weights")
WEIGHTS_DIR = "/weights"
MAX_IMAGE_BYTES = 20 * 1024 * 1024          # 20 MB upload cap
MAX_ASSET_BYTES = 64 * 1024 * 1024          # 64 MB per downloaded asset cap

# Cross-origin access for the React dev server / any deployed static frontend.
# The Vite dev server proxies /generate-3d, so this is mainly for direct calls.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

ASSET_BASE_URL = os.environ.get("ASSET_BASE_URL", "https://api.sketchfab.com/v3")
ASSET_API_TOKEN = os.environ.get("ASSET_API_TOKEN")
# How many downloadable search candidates to try per furniture piece before
# falling back to a procedural mesh — makes "matches the extraction" far more
# likely than the old single top-hit.
ASSET_TOP_K = int(os.environ.get("ASSET_TOP_K", "3"))

# Downloaded GLBs are cached by search query so repeat runs skip the network.
# A Modal volume named "dolgen-assets" is mounted here when it exists; locally
# (or if the volume is absent) this just falls back to a tmp dir.
ASSET_CACHE_DIR = os.environ.get("ASSET_CACHE_DIR", "/cache/assets")
try:
    os.makedirs(ASSET_CACHE_DIR, exist_ok=True)
except OSError:  # read-only / nonexistent mount (e.g. local run) — use tmp
    ASSET_CACHE_DIR = os.path.join("/tmp", "dolgen-assets")
    os.makedirs(ASSET_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Gemini extraction schema
# ---------------------------------------------------------------------------

class FloorElement(BaseModel):
    """One extracted floorplan element. box_2d is [ymin, xmin, ymax, xmax]
    on a 0-1000 normalized image coordinate system ((0,0) = top-left)."""

    element_type: Literal["wall", "door", "window", "furniture"]
    furniture_class: Optional[str] = None            # e.g. "bed", "sofa"; null if not furniture
    asset_search_query: Optional[str] = None         # 3-5 word 3D-asset query; null if not furniture
    box_2d: list[float]                              # kept float here; rounded/clamped downstream

    @field_validator("box_2d")
    @classmethod
    def _box_has_four_sides(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("box_2d must have exactly 4 numbers [ymin, xmin, ymax, xmax]")
        return v


GEMINI_PROMPT = """You are an architectural floorplan understanding engine.

Extract every structural element and every piece of furniture visible in this floorplan image,
and return ONLY a JSON object matching the provided schema: a top-level object with an
"elements" array (and optionally "room_colors").

Rules:
- box_2d = [ymin, xmin, ymax, xmax] as integers normalized to 0-1000, where (0,0) is the
  TOP-LEFT of the image. Boxes must be axis-aligned and tight around the element.
- element_type "wall": every solid wall segment (exterior and interior). Represent each straight
  wall segment as its own thin box. Walls should join to form closed rooms where possible.
- element_type "door": every door opening through a wall; the box covers the opening rectangle.
- element_type "window": every window opening through a wall; the box covers the opening rectangle.
- element_type "furniture": movable objects only (bed, sofa, table, chair, desk, wardrobe, bathtub,
  toilet, sink, stove, fridge, ...). The box is the object's top-down floor footprint. Also set:
    * furniture_class      -> a short lower-case class name, e.g. "bed", "sofa", "dining_table"
    * asset_search_query   -> a specific natural-language search query (type + material + color +
                              style) that would find a good 3D model of THIS object in a 3D asset
                              library, e.g. "modern grey fabric sectional sofa", "oak queen bed frame",
                              "white ceramic pedestal sink"
- asset_search_query should describe the SPECIFIC object, not a generic category: include
  type + material + color + style, e.g. "modern grey fabric sectional sofa", "oak queen bed frame",
  "white ceramic pedestal sink". This query is used verbatim to fetch a matching 3D model from
  a web asset library, so make it visually faithful to what is drawn.
- furniture_class and asset_search_query MUST be null when element_type is not "furniture".
- If a room's fill color or a clear label implies a wall/room color, add a top-level "room_colors"
  object mapping a short room name to a hex color string, e.g. {"bedroom": "#c8d6e8"}. If nothing
  is implied, omit it or return an empty object.
- Ignore text labels, dimension lines, arrows, scale bars and north symbols.
- Respond with a single JSON object only: {"elements": [...], "room_colors": {...}}. Wrap the
  element array in the top-level "elements" key; "room_colors" may be omitted or empty.

Be thorough: missing walls or furniture makes the 3D model wrong."""


async def extract_elements(image_bytes: bytes, mime_type: str) -> list[FloorElement]:
    """Call Kimi K3 (vision) on the floorplan; return validated elements."""
    try:
        client = AsyncOpenAI(base_url=KIMI_BASE_URL, api_key=_kimi_api_key())
    except Exception as exc:  # pragma: no cover - depends on runtime config
        raise HTTPException(
            status_code=500,
            detail=f"Kimi client could not be created (KIMI_BASE_URL / proxy token): {exc}",
        )

    b64_image = base64.b64encode(image_bytes).decode("ascii")
    try:
        completion = await client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                        },
                        {"type": "text", "text": GEMINI_PROMPT},
                    ],
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            # Kimi K3 always reasons (it cannot disable thinking); "low" is the
            # cheapest setting the endpoint accepts.
            extra_body={"reasoning_effort": "low"},
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Kimi analysis failed: {exc}")

    text = (completion.choices[0].message.content or "").strip() if completion.choices else ""
    elements: list[Any] = []
    if text:
        try:
            obj = json.loads(text)
            # JSON mode returns an object; the element array lives under "elements".
            if isinstance(obj, dict):
                obj = obj.get("elements", [])
            if isinstance(obj, list):
                elements = obj
        except (json.JSONDecodeError, AttributeError):
            elements = []
    parsed = []
    for o in elements:
        try:
            parsed.append(FloorElement.model_validate(o))
        except Exception:
            continue
    if not parsed:
        raise HTTPException(
            status_code=422,
            detail="Kimi returned no extractable elements for this floorplan image.",
        )
    return parsed


def normalize_elements(parsed: list[FloorElement]) -> list[dict[str, Any]]:
    """Round/clamp boxes into integer 0-1000 space and drop degenerate ones."""
    out: list[dict[str, Any]] = []
    for el in parsed:
        ymin, xmin, ymax, xmax = el.box_2d
        ymin, ymax = sorted((min(1000.0, max(0.0, ymin)), min(1000.0, max(0.0, ymax))))
        xmin, xmax = sorted((min(1000.0, max(0.0, xmin)), min(1000.0, max(0.0, xmax))))
        box = [int(round(v)) for v in (ymin, xmin, ymax, xmax)]
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            logger.info("Dropping degenerate box %s for %s", box, el.element_type)
            continue
        out.append(
            {
                "element_type": el.element_type,
                "furniture_class": el.furniture_class,
                "asset_search_query": el.asset_search_query,
                "box_2d": box,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Geometry helpers (single shared coordinate system, one source of truth)
#
# Image space:  (ymin..ymax) x (xmin..xmax) in 0-1000, origin top-left.
# 3D space:     x_m = xmin*SCALE .. xmax*SCALE   (east)
#               y_m = -(ymax)*SCALE .. -(ymin)*SCALE  (south of image top, inverted
#                   so the dollhouse reads the same way round as the image)
#               z_m = 0 (floor) .. +WALL_HEIGHT
#
# Every mesh in the scene is built by scale_mesh() from an image-space box plus
# (z_min, z_max) — walls, floor, door/window cutter blocks and furniture
# placement footprints all share this exact transform, so nothing drifts.
# ---------------------------------------------------------------------------

def scale_mesh(box: list[int], z_min: float, z_max: float) -> trimesh.Trimesh:
    """Create a solid box mesh occupying `box` (image units) between z_min and z_max.

    This is the single transform used by every element in the scene.
    """
    ymin, xmin, ymax, xmax = box
    x_lo, x_hi = xmin * SCALE, xmax * SCALE
    y_lo, y_hi = -ymax * SCALE, -ymin * SCALE
    mesh = trimesh.creation.box(
        extents=(max(x_hi - x_lo, 1e-4), max(y_hi - y_lo, 1e-4), max(z_max - z_min, 1e-4))
    )
    mesh.apply_translation(((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_min + z_max) / 2))
    return mesh


def box_center_xy_meters(box: list[int]) -> tuple[float, float]:
    ymin, xmin, ymax, xmax = box
    return ((xmin + xmax) / 2 * SCALE, -(ymin + ymax) / 2 * SCALE)


def opening_hparams(box: list[int], kind: str) -> tuple[float, float]:
    """(z_min, z_max) of a door/window cutter.

    The 2D footprint of an opening says nothing about its real height, so use
    standard architectural dimensions: a 2.1 m door from the floor, and a window
    with a 0.9 m sill and 2.1 m head. Both are clamped to the wall height.
    """
    if kind == "window":
        return min(0.9, WALL_HEIGHT * 0.5), min(2.1, WALL_HEIGHT)
    return 0.0, min(2.1, WALL_HEIGHT)


def pbr_material(color_rgba: tuple[int, int, int, int], name: str) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=np.array(color_rgba, dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )


def mesh_with_material(mesh: trimesh.Trimesh, mat: trimesh.visual.material.PBRMaterial) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh


def boolean_difference(wall: trimesh.Trimesh, cutters: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Subtract `cutters` from `wall` with the manifold engine.

    Robustness: if a boolean fails (near-degenerate coplanar faces), retry once
    with the cutters inflated by a hair; if that still fails, keep the wall whole
    rather than crash the request — worst case, an opening is not cut.
    """
    if not cutters:
        return wall
    try:
        result = wall.difference(cutters, engine="manifold")
        if result is not None and not result.is_empty:
            return result
    except Exception as exc:
        logger.warning("Boolean difference failed (%s); retrying with inflated cutters", exc)
    try:
        inflated = [c.copy() for c in cutters]
        for c in inflated:
            c.apply_scale(1.0005)
        result = wall.difference(inflated, engine="manifold")
        if result is not None and not result.is_empty:
            return result
    except Exception as exc:
        logger.warning("Boolean difference retry failed (%s); using un-cut wall", exc)
    return wall


def build_shell(
    elements: list[dict[str, Any]],
    wall_material: Optional[trimesh.visual.material.PBRMaterial] = None,
    floor_material: Optional[trimesh.visual.material.PBRMaterial] = None,
) -> tuple[trimesh.Scene, list[str]]:
    """Build floor + walls with door/window openings. Returns (scene, log lines)."""
    logs: list[str] = []
    scene = trimesh.Scene()

    white = wall_material or pbr_material((238, 238, 234, 255), "wall_white")
    wood = floor_material or pbr_material((196, 154, 108, 255), "floor_light_wood")

    # Floor spans the whole image footprint, slab extends below z=0.
    floor = scale_mesh([0, 0, 1000, 1000], -FLOOR_THICKNESS, 0.0)
    scene.add_geometry(mesh_with_material(floor, wood), node_name="floor", geom_name="floor")

    walls = [e for e in elements if e["element_type"] == "wall"]
    doors = [e for e in elements if e["element_type"] == "door"]
    windows = [e for e in elements if e["element_type"] == "window"]
    logs.append(f"Extraction: {len(walls)} walls, {len(doors)} doors, {len(windows)} windows.")

    def wall_mesh_from_box(box: list[int]) -> trimesh.Trimesh:
        """Build a wall of *exactly* WALL_THICKNESS, centered on the box's thin axis.

        Walls returned by Gemini are often fatter than real walls (it outlines the
        whole stroke). We keep the box's long-axis extent but force the thin axis to
        WALL_THICKNESS so rooms stay the right size and the enclosing scale holds.
        """
        ymin, xmin, ymax, xmax = box
        thin = 0 if (xmax - xmin) < (ymax - ymin) else 1  # thin axis: 0 = x, 1 = y
        half_units = (WALL_THICKNESS / SCALE) / 2        # wall half-thickness in image units
        if thin == 0:  # x thin -> wall runs along Y (north-south)
            x_mid = (xmin + xmax) / 2
            new_box = [ymin, int(round(x_mid - half_units)), ymax, int(round(x_mid + half_units))]
        else:          # y thin -> wall runs along X (east-west)
            y_mid = (ymin + ymax) / 2
            new_box = [int(round(y_mid - half_units)), xmin, int(round(y_mid + half_units)), xmax]
        # Guard against a degenerate rebuild (e.g. a near-point box).
        if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
            new_box = box
        return scale_mesh(new_box, 0.0, WALL_HEIGHT)

    def opening_cutters_through(wall_mesh: trimesh.Trimesh, op: dict[str, Any], kind: str) -> Optional[trimesh.Trimesh]:
        """Build a cutter for a door/window through THIS wall.

        The opening only cuts this wall if its 2D footprint actually sits on the
        wall: the two must overlap along the wall's long axis, and the opening's
        center must lie on (within a tolerance of) the wall's thin axis. Without
        this gate every window/door in the plan was punching a hole through every
        wall it geometrically crossed — e.g. a window on the north wall also
        blowing a full-height gap through the parallel south wall.
        """
        oymin, oxmin, oymax, oxmax = op["box_2d"]
        wall_ext = wall_mesh.extents          # meters
        thin_axis = 0 if wall_ext[0] < wall_ext[1] else 1
        long_axis = 1 - thin_axis
        z_min, z_max = opening_hparams(op["box_2d"], kind)

        # The opening's 2D box as a physical rectangle (matching scale_mesh's y-flip).
        op_x_lo, op_x_hi = oxmin * SCALE, oxmax * SCALE
        op_y_lo, op_y_hi = -oymax * SCALE, -oymin * SCALE
        op_lo = (op_x_lo, op_y_lo)
        op_hi = (op_x_hi, op_y_hi)

        # 1) Long-axis overlap: the opening must share a real span with this wall.
        w_long_lo = wall_mesh.bounds[0, long_axis]
        w_long_hi = wall_mesh.bounds[1, long_axis]
        ov_lo = max(op_lo[long_axis], w_long_lo)
        ov_hi = min(op_hi[long_axis], w_long_hi)
        if ov_hi - ov_lo <= 1e-3:
            return None  # opening doesn't reach this wall along its length

        # 2) Thin-axis proximity: the opening's center must sit on the wall.
        #    Tolerance = wall half-thickness + a little slack for a loosely drawn box.
        w_thin_lo = wall_mesh.bounds[0, thin_axis]
        w_thin_hi = wall_mesh.bounds[1, thin_axis]
        op_thin_mid = (op_lo[thin_axis] + op_hi[thin_axis]) / 2
        thin_slack = (w_thin_hi - w_thin_lo) / 2 + 0.35  # ~wall half-thickness + 0.35 m
        if not (w_thin_lo - thin_slack) <= op_thin_mid <= (w_thin_hi + thin_slack):
            return None  # opening is on a different (e.g. parallel) wall

        # Hole width: clamp the opening's long-axis span to the wall segment so a
        # box drawn wider than the wall can't over-cut into the room.
        op_long_lo, op_long_hi = ov_lo, ov_hi

        # Cutter thin span = wall thin span + margin, so it passes fully through.
        margin = 0.05
        lo = [0.0, 0.0, z_min]
        hi = [0.0, 0.0, z_max]
        lo[long_axis], hi[long_axis] = op_long_lo, op_long_hi
        lo[thin_axis] = w_thin_lo - margin
        hi[thin_axis] = w_thin_hi + margin
        extents = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        if min(extents) <= 1e-4:
            return None
        cutter = trimesh.creation.box(extents=extents)
        cutter.apply_translation(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2))
        return cutter

    # Keep compact local lists for the lambdas below.
    for i, w in enumerate(walls):
        wall_mesh = wall_mesh_from_box(w["box_2d"])
        cutters: list[trimesh.Trimesh] = []
        for op in doors:
            c = opening_cutters_through(wall_mesh, op, "door")
            if c is not None:
                cutters.append(c)
        for op in windows:
            c = opening_cutters_through(wall_mesh, op, "window")
            if c is not None:
                cutters.append(c)
        wall_mesh = boolean_difference(wall_mesh, cutters)
        name = f"wall_{i}"
        scene.add_geometry(mesh_with_material(wall_mesh, white), node_name=name, geom_name=name)

    return scene, logs


# ---------------------------------------------------------------------------
# Dynamic asset fetching (Sketchfab-compatible) — graceful skip on any failure
# ---------------------------------------------------------------------------

def _cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:24] + ".glb"


def _looks_like_glb(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"glTF"


def _glb_parses(data: bytes) -> bool:
    """Cheap validity gate: a candidate only 'matches' if trimesh can load it
    and it has real geometry — otherwise we keep looking."""
    try:
        mesh = _as_one_mesh(trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False))
        return bool(np.all(np.isfinite(mesh.extents)) and mesh.extents.max() > 1e-5)
    except Exception:
        return False


async def _fetch_one_asset(
    client: httpx.AsyncClient, query: str
) -> tuple[Optional[bytes], str]:
    """Fetch a furniture asset from the web asset library that *matches* `query`.

    Instead of taking the first search hit, walk the top-`ASSET_TOP_K`
    downloadable candidates and return the first whose .glb both downloads and
    actually parses — so the placed model is far more likely to resemble what
    Gemini extracted. Results are cached on disk keyed by query.
    """
    key = _cache_key(query)
    cached = os.path.join(ASSET_CACHE_DIR, key)
    if os.path.exists(cached):
        try:
            with open(cached, "rb") as fh:
                return fh.read(), "cache"
        except OSError:
            pass  # fall through to the network

    headers = {"Authorization": f"Bearer {ASSET_API_TOKEN}"} if ASSET_API_TOKEN else {}
    try:
        search = await client.get(
            f"{ASSET_BASE_URL}/search",
            params={"type": "models", "q": query, "downloadable": "true", "count": str(ASSET_TOP_K)},
            headers=headers,
        )
        search.raise_for_status()
        results = (search.json().get("results") or [])[: max(ASSET_TOP_K, 1)]
        if not results:
            return None, f"no downloadable results for '{query}'"

        last_err = "no usable candidate"
        for hit in results:
            uid = hit.get("uid") if isinstance(hit, dict) else None
            if not uid:
                continue
            try:
                dl = await client.post(f"{ASSET_BASE_URL}/models/{uid}/download", headers=headers)
                dl.raise_for_status()
                payload = dl.json()
                fmt = payload.get("glb") or payload.get("gltf") or {}
                url = fmt.get("url") if isinstance(fmt, dict) else None
                if not isinstance(fmt, dict) or not url:
                    url = None
                    for entry in payload if isinstance(payload, list) else [payload]:
                        if isinstance(entry, dict) and entry.get("url"):
                            url = entry["url"]
                            break
                if not url:
                    last_err = "no direct .glb/.gltf url in download response"
                    continue

                resp = await client.get(url, headers=headers, follow_redirects=True)
                resp.raise_for_status()
                data = resp.content
                if len(data) > MAX_ASSET_BYTES:
                    last_err = f"asset too large ({len(data)} bytes)"
                    continue
                if _looks_like_glb(data) and _glb_parses(data):
                    try:
                        with open(cached, "wb") as fh:
                            fh.write(data)
                    except OSError:
                        pass
                    return data, "ok"
                last_err = "candidate was not a loadable GLB"
            except httpx.HTTPStatusError as exc:
                last_err = f"download HTTP {exc.response.status_code}"
            except Exception as exc:  # keep trying the next candidate
                last_err = f"{type(exc).__name__}: {exc}"
        return None, f"all {len(results)} candidates failed ({last_err})"
    except httpx.HTTPStatusError as exc:
        return None, f"asset API HTTP {exc.response.status_code}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def download_assets(
    elements: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], bytes]], list[str]]:
    """Download every furniture element's asset concurrently. Skips failures."""
    furniture = [e for e in elements if e["element_type"] == "furniture" and e["asset_search_query"]]
    logs: list[str] = [f"Fetching {len(furniture)} furniture assets from {ASSET_BASE_URL}..."]
    ok: list[tuple[dict[str, Any], bytes]] = []
    if not furniture:
        return ok, logs

    async with httpx.AsyncClient(timeout=20.0) as client:
        results = await asyncio.gather(
            *( _fetch_one_asset(client, e["asset_search_query"]) for e in furniture )
        )
    for el, (data, status) in zip(furniture, results):
        name = el.get("furniture_class") or el["asset_search_query"]
        if data is None:
            logs.append(f"Asset download failed for '{name}' ({status}) — skipping furniture.")
        else:
            logs.append(f"Downloaded asset for '{name}' ({len(data)//1024} KB).")
            ok.append((el, data))
    return ok, logs


# ---------------------------------------------------------------------------
# AI-generated furniture models (Gemini image model "nano banana pro")
#
# For each furniture piece we crop its footprint out of the uploaded floorplan,
# ask the Gemini image model to render that piece as a clean top-down product
# photo, texture-map the image onto a procedural mesh of the piece, and export
# it as a self-contained .glb — which then flows through the exact same
# placement path as a downloaded asset (normalize_furniture_to_box). Any failure
# yields None and the caller falls back to the untextured procedural mesh.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Open-weights furniture renderer (FLUX.1 Kontext [dev] on Modal GPU)
#
# Active only when AI_IMAGE_BACKEND=openweights. The class holds the pipeline
# in a GPU container; containers scale to zero when idle (scaledown_window=0)
# so nothing is paid while unused — the cost is that every cold start pays
# the model-load time (seconds once WEIGHTS_DIR is warm, minutes on the very
# first pull). Every render failure returns None and the caller falls back to
# the Gemini image model, then to a procedural mesh, per the usual layering.
# ---------------------------------------------------------------------------

OPENWEIGHTS_PROMPT = """Turn this 2D architectural floorplan symbol (a {furniture_class}) into a
photorealistic, perfectly top-down (orthographic plan view) product photo of the real 3D piece.
Camera directly overhead, looking straight down. Keep the piece's orientation in the frame
exactly as drawn. Center it, filling most of the frame with a small even margin. Pure solid
white background (#ffffff), soft even studio lighting, subtle contact shadow. No text, labels,
dimension lines, borders, floor lines, or other objects.{style_hint}"""


@app.cls(
    image=image,
    gpu="L40S",
    volumes={WEIGHTS_DIR: modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)},
    secrets=[modal.Secret.from_name("gemini-secret")],  # HF_TOKEN can live here if the repo is gated
    scaledown_window=2,      # cold-only: scale to zero right after each call (Modal min is 2s)
    timeout=900,             # generous for first-ever weights download
    memory=32768,
)
class FurnitureRenderer:
    """Holds a loaded FLUX.1 Kontext pipeline for the lifetime of one warm container."""

    @modal.enter()
    def load_pipeline(self) -> None:
        import torch
        from diffusers import FluxKontextPipeline

        logger.info("Loading %s onto GPU (cold start)...", OPENWEIGHTS_MODEL)
        self.pipe = FluxKontextPipeline.from_pretrained(
            OPENWEIGHTS_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=WEIGHTS_DIR,
        )
        self.pipe.to("cuda")
        logger.info("Furniture renderer ready.")

    @modal.method()
    def render(self, crop_png: bytes, furniture_class: str, style_hint: str) -> bytes | None:
        """One top-down product render of the furniture piece, as PNG bytes."""
        import io as _io

        from PIL import Image

        try:
            init_image = Image.open(_io.BytesIO(crop_png)).convert("RGB")
            prompt = OPENWEIGHTS_PROMPT.format(
                furniture_class=furniture_class or "furniture",
                style_hint=f"\nFor reference, the piece looks like: {style_hint}." if style_hint else "",
            )
            out = self.pipe(
                image=init_image,
                prompt=prompt,
                guidance_scale=2.5,
                num_inference_steps=28,
            )
            img = out.images[0]
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            logger.info("Open-weights render failed for '%s': %s", furniture_class, exc)
            return None


AI_FURNITURE_PROMPT = """You are given a crop of a 2D architectural floorplan showing one piece of furniture
(a {furniture_class}).

Render THIS object as a photorealistic, perfectly top-down (orthographic plan view) product
image of the real 3D piece it represents. The camera is directly overhead, looking straight
down.

Requirements:
- The piece faces the same direction as in the floorplan (same orientation in the frame).
- The piece is centered and fills most of the frame, with a small even margin on all sides.
- Background: pure solid white (#ffffff), nothing else in the image.
- Soft, even studio lighting with a subtle contact shadow.
- No text, labels, dimension lines, borders, floor lines, or other objects."""


def _image_response_may_retry(exc: Exception) -> bool:
    """429/5xx (and UNAVAILABLE/RESOURCE_EXHAUSTED) are worth one retry."""
    if GenAIClientError is not None and isinstance(exc, GenAIClientError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        return code in (408, 409, 425, 429, 500, 502, 503, 504)
    return False


def _decode_image_bytes(part: Any) -> bytes | None:
    """Best-effort extraction of PNG/JPEG bytes from a google-genai response part."""
    try:
        img = part.as_image()
        if img is not None:
            if getattr(img, "image_bytes", None):
                return img.image_bytes
            if getattr(img, "_loaded_image", None) is not None:
                buf = io.BytesIO()
                img._loaded_image.save(buf, format="PNG")
                return buf.getvalue()
    except Exception:
        pass
    inline = getattr(part, "inline_data", None)
    if inline is not None and getattr(inline, "data", None):
        data = inline.data
        if isinstance(data, str):  # some SDK versions hand back base64 text
            import base64

            try:
                return base64.b64decode(data)
            except Exception:
                return None
        return bytes(data)
    return None


def _crop_pad_floorplan(
    image_bytes: bytes, box: list[int], pad_frac: float = 0.15
) -> bytes | None:
    """Crop a furniture footprint (0-1000 image units) out of the floorplan PNG."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow is in the Modal image
        return None
    ymin, xmin, ymax, xmax = box
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    W, H = im.size
    x0 = xmin / 1000.0 * W
    x1 = xmax / 1000.0 * W
    y0 = ymin / 1000.0 * H
    y1 = ymax / 1000.0 * H
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return None
    px, py = w * pad_frac, h * pad_frac
    crop = im.crop((
        max(0, int(x0 - px)),
        max(0, int(y0 - py)),
        min(W, int(x1 + px)),
        min(H, int(y1 + py)),
    ))
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(crop, ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


async def _gen_furniture_view_openweights(crop_png: bytes, el: dict[str, Any]) -> bytes | None:
    """Render via the GPU-resident FLUX.1 Kontext pipeline. None on any failure
    (the caller then tries Gemini, then a procedural mesh)."""
    try:
        renderer = FurnitureRenderer()
        return await renderer.render.remote.aio(
            crop_png,
            el.get("furniture_class") or "furniture",
            el.get("asset_search_query") or "",
        )
    except Exception as exc:
        logger.info("Open-weights backend unavailable: %s", exc)
        return None


async def _gen_furniture_view(
    client: genai.Client, crop_png: bytes, el: dict[str, Any]
) -> bytes | None:
    """One render of the furniture piece (top-down view).

    Backend chosen by AI_IMAGE_BACKEND: "openweights" tries the GPU pipeline
    first and falls back to the Gemini image model on failure; "gemini" (the
    default) goes straight to the Gemini image model.
    """
    if AI_IMAGE_BACKEND == "openweights":
        data = await _gen_furniture_view_openweights(crop_png, el)
        if data is not None:
            return data
        logger.info(
            "Falling back to Gemini image model for '%s'.",
            el.get("furniture_class") or "furniture",
        )
    elif AI_IMAGE_BACKEND != "gemini":
        logger.warning("Unknown AI_IMAGE_BACKEND '%s' — using gemini.", AI_IMAGE_BACKEND)

    prompt = AI_FURNITURE_PROMPT.format(
        furniture_class=el.get("furniture_class") or "furniture"
    )
    if el.get("asset_search_query"):
        prompt += f"\nFor reference, the piece looks like: {el['asset_search_query']}."
    contents = [
        genai_types.Part.from_bytes(data=crop_png, mime_type="image/png"),
        prompt,
    ]
    cfg_kwargs: dict[str, Any] = {"temperature": 1.0}
    try:  # IMAGE-only modality where the SDK exposes it
        cfg_kwargs["response_modalities"] = ["IMAGE"]
    except Exception:
        pass
    try:
        cfg_kwargs["image_config"] = genai_types.ImageConfig(aspect_ratio="1:1")
    except Exception:
        pass
    config = genai_types.GenerateContentConfig(**cfg_kwargs)

    for attempt in (1, 2):
        try:
            resp = await client.aio.models.generate_content(
                model=GEMINI_IMAGE_MODEL, contents=contents, config=config
            )
        except Exception as exc:
            if attempt == 1 and _image_response_may_retry(exc):
                await asyncio.sleep(2.0)
                continue
            logger.info("Gemini image generation failed: %s", exc)
            return None
        try:
            for part in resp.parts or []:
                data = _decode_image_bytes(part)
                if data:
                    return data
        except Exception:
            pass
        return None
    return None


def _top_texture(mesh: trimesh.Trimesh, image_bytes: bytes) -> None:
    """Planar-project `image_bytes` (a top-down render) onto the top of `mesh`.

    UV u/v run along the mesh's x/y extents, so after placement the texture is
    aligned with the furniture's footprint (both come from the same 2D box).
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    b = mesh.bounds
    span_x = max(float(b[1][0] - b[0][0]), 1e-6)
    span_y = max(float(b[1][1] - b[0][1]), 1e-6)
    uv = np.column_stack([
        (mesh.vertices[:, 0] - b[0][0]) / span_x,
        (mesh.vertices[:, 1] - b[0][1]) / span_y,
    ])
    mat = trimesh.visual.material.PBRMaterial(
        name="ai_top",
        image=img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)


def build_ai_furniture_glb(
    furniture_class: str, box: list[int], view_png: bytes
) -> bytes | None:
    """Textured procedural piece -> standalone .glb, or None on any failure."""
    try:
        mesh = build_procedural_furniture(furniture_class or "furniture", box)
        _top_texture(mesh, view_png)
        data = mesh.export(file_type="glb")
        return data if _looks_like_glb(data) else None
    except Exception as exc:
        logger.info("AI furniture glb build failed: %s", exc)
        return None


async def generate_ai_furniture_assets(
    elements: list[dict[str, Any]], image_bytes: bytes
) -> tuple[list[tuple[dict[str, Any], bytes]], list[str]]:
    """Generate .glb models for furniture pieces via the Gemini image model."""
    furniture = [e for e in elements if e["element_type"] == "furniture"]
    logs = [
        f"Generating AI models for {len(furniture)} furniture piece(s) "
        f"with {GEMINI_IMAGE_MODEL}..."
    ]
    if not furniture:
        return [], logs
    try:
        client = genai.Client()
    except Exception as exc:
        return [], [
            f"Gemini client unavailable for image generation ({exc}) — no AI furniture."
        ]

    async def one(el: dict[str, Any]) -> bytes | None:
        crop = _crop_pad_floorplan(image_bytes, el["box_2d"])
        if crop is None:
            return None
        view = await _gen_furniture_view(client, crop, el)
        if view is None:
            return None
        return build_ai_furniture_glb(el.get("furniture_class") or "furniture", el["box_2d"], view)

    results = await asyncio.gather(*(one(el) for el in furniture))
    ok: list[tuple[dict[str, Any], bytes]] = []
    for el, data in zip(furniture, results, strict=True):
        name = el.get("furniture_class") or "furniture"
        if data is None:
            logs.append(
                f"AI model generation failed for '{name}' — procedural fallback will be used."
            )
        else:
            logs.append(f"AI-generated model for '{name}' ({len(data)//1024} KB).")
            ok.append((el, data))
    return ok, logs


# ---------------------------------------------------------------------------
# Furniture normalization / placement
# ---------------------------------------------------------------------------

def _as_one_mesh(loaded: Any) -> trimesh.Trimesh:
    """Coerce a trimesh.load() result to a single Trimesh (concatenating scenes)."""
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("scene contains no triangle meshes")
        if len(meshes) == 1:
            return meshes[0]
        return trimesh.util.concatenate(meshes)
    raise ValueError(f"unsupported loaded object: {type(loaded)}")


def normalize_furniture_to_box(mesh: trimesh.Trimesh, box: list[int]) -> trimesh.Trimesh:
    """Fit `mesh` into the footprint described by `box` (image units), on the floor.

    Uniform scale so (x-extent, y-extent) never exceeds the footprint; z scaled the
    same. The mesh's minimum XY corner is placed at the box's minimum corner.
    """
    ymin, xmin, ymax, xmax = box
    footprint_x = (xmax - xmin) * SCALE
    footprint_y = (ymax - ymin) * SCALE

    ext = mesh.extents  # current x/y/z extents
    if not np.all(np.isfinite(ext)) or ext[:2].min() <= 1e-6:
        raise ValueError("degenerate asset geometry")

    s = min(footprint_x / ext[0], footprint_y / ext[1])
    mesh.apply_scale(s)

    # Re-ground: shift so the min corner sits exactly at the box min corner, z on floor.
    mesh.apply_translation((-mesh.bounds[0][0], -mesh.bounds[0][1], -mesh.bounds[0][2]))
    mesh.apply_translation((xmin * SCALE, -ymax * SCALE, 0.0))
    return mesh


# ---------------------------------------------------------------------------
# Procedural furniture fallback (used when the web asset library has no match)
# ---------------------------------------------------------------------------

def _part(extents: tuple[float, float, float], offset: tuple[float, float, float],
          mat: trimesh.visual.material.PBRMaterial) -> trimesh.Trimesh:
    """A single box primitive at a local offset (min-corner anchored at origin)."""
    m = trimesh.creation.box(extents=extents)
    m.apply_translation((offset[0] + extents[0] / 2, offset[1] + extents[1] / 2, offset[2] + extents[2] / 2))
    return mesh_with_material(m, mat)


def _concat(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def _proc_bed(w: float, d: float, fab, wood) -> trimesh.Trimesh:
    frame_h, matt_h, head_h, pillow_h = 0.30, 0.22, 0.95, 0.12
    return _concat([
        _part((w, d, frame_h), (0, 0, 0), wood),                                   # frame
        _part((w, d * 0.12, head_h), (0, d * 0.88, 0), wood),                      # headboard
        _part((w * 0.94, d * 0.82, matt_h), (w * 0.03, d * 0.03, frame_h), fab),   # mattress
        _part((w * 0.40, d * 0.20, pillow_h), (w * 0.08, d * 0.66, frame_h + matt_h), fab),  # pillows
        _part((w * 0.40, d * 0.20, pillow_h), (w * 0.52, d * 0.66, frame_h + matt_h), fab),
    ])


def _proc_sofa(w: float, d: float, fab, _wood) -> trimesh.Trimesh:
    base_h, back_h, arm_h, arm_w = 0.40, 0.85, 0.60, min(0.18, w * 0.12)
    seat_d = d - arm_w
    return _concat([
        _part((w, seat_d, base_h), (0, 0, 0), fab),                                # seat base
        _part((w, d - seat_d, back_h), (0, seat_d, 0), fab),                       # backrest
        _part((arm_w, d, arm_h), (0, 0, 0), fab),                                  # left arm
        _part((arm_w, d, arm_h), (w - arm_w, 0, 0), fab),                          # right arm
        _part((w - 2 * arm_w, seat_d * 0.9, 0.12), (arm_w, 0, base_h), fab),       # seat cushion
    ])


def _proc_table(w: float, d: float, wood, _fab) -> trimesh.Trimesh:
    top_h, top_t, leg = 0.74, 0.04, 0.06
    inset = leg * 0.6
    leg_h = top_h - top_t
    parts = [_part((w, d, top_t), (0, 0, leg_h), wood)]
    for ox in (inset, w - inset - leg):
        for oy in (inset, d - inset - leg):
            parts.append(_part((leg, leg, leg_h), (ox, oy, 0), wood))
    return _concat(parts)


def _proc_chair(w: float, d: float, wood, _fab) -> trimesh.Trimesh:
    seat_h, seat_t, back_h, leg = 0.45, 0.04, 0.90, 0.045
    parts = [
        _part((w, d, seat_t), (0, 0, seat_h - seat_t), wood),                      # seat
        _part((w, seat_t, back_h - seat_h), (0, d - seat_t, seat_h), wood),        # backrest
    ]
    for ox in (0.0, w - leg):
        for oy in (0.0, d - leg):
            parts.append(_part((leg, leg, seat_h), (ox, oy, 0), wood))
    return _concat(parts)


def _proc_desk(w: float, d: float, wood, _fab) -> trimesh.Trimesh:
    top_h, top_t = 0.74, 0.04
    return _concat([
        _part((w, d, top_t), (0, 0, top_h - top_t), wood),                         # top
        _part((top_t, d, top_h), (0, 0, 0), wood),                                 # left panel
        _part((top_t, d, top_h), (w - top_t, 0, 0), wood),                         # right panel
    ])


def _proc_wardrobe(w: float, d: float, wood, _fab) -> trimesh.Trimesh:
    h = min(2.0, WALL_HEIGHT * 0.8)
    handle = _part((0.03, 0.03, 0.18), (w / 2 - 0.015, -0.02, h / 2), pbr_material((90, 90, 95, 255), "handle"))
    return _concat([_part((w, d, h), (0, 0, 0), wood), handle])


def _proc_bathtub(w: float, d: float, ceram, _fab) -> trimesh.Trimesh:
    h, rim = 0.55, 0.08
    wall = 0.06
    parts = [
        _part((w, d, 0.06), (0, 0, 0), ceram),                                     # base
        _part((w, rim, h), (0, 0, 0), ceram),                                      # near side
        _part((w, rim, h), (0, d - rim, 0), ceram),                                # far side
        _part((wall, d - 2 * rim, h), (0, rim, 0), ceram),                         # ends
        _part((wall, d - 2 * rim, h), (w - wall, rim, 0), ceram),
    ]
    return _concat(parts)


def _proc_toilet(w: float, d: float, ceram, _fab) -> trimesh.Trimesh:
    return _concat([
        _part((w * 0.7, d * 0.6, 0.42), (w * 0.15, 0, 0), ceram),                  # bowl
        _part((w, d * 0.32, 0.75), (0, d * 0.66, 0), ceram),                       # cistern
    ])


def _proc_sink(w: float, d: float, ceram, _fab) -> trimesh.Trimesh:
    return _concat([
        _part((w * 0.16, w * 0.16, 0.78), (w / 2 - w * 0.08, d / 2 - w * 0.08, 0), ceram),  # pedestal
        _part((w, d, 0.12), (0, 0, 0.78), ceram),                                  # basin
    ])


def _proc_stove(w: float, d: float, metal, _fab) -> trimesh.Trimesh:
    h = 0.9
    body = _part((w, d, h), (0, 0, 0), metal)
    dark = pbr_material((35, 35, 38, 255), "burner")
    r = min(w, d) * 0.18
    parts = [body]
    for ox, oy in ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)):
        c = trimesh.creation.cylinder(radius=r, height=0.02, sections=20)
        c.apply_translation((w * ox, d * oy, h + 0.01))
        parts.append(mesh_with_material(c, dark))
    return _concat(parts)


def _proc_fridge(w: float, d: float, metal, _fab) -> trimesh.Trimesh:
    h = 1.8
    body = _part((w, d, h), (0, 0, 0), metal)
    seam = _part((w + 0.005, d + 0.005, 0.01), (-0.0025, -0.0025, h * 0.62), pbr_material((70, 70, 74, 255), "seam"))
    handle = _part((0.03, 0.03, 0.5), (w * 0.06, -0.02, h * 0.65), pbr_material((70, 70, 74, 255), "handle"))
    return _concat([body, seam, handle])


def _proc_generic(w: float, d: float, a, _b) -> trimesh.Trimesh:
    return _part((w, d, 0.5), (0, 0, 0), a)


def build_procedural_furniture(furniture_class: str, box: list[int]) -> trimesh.Trimesh:
    """Synthesize a simple recognizable mesh for `furniture_class` that fits the
    footprint `box`. Used when no downloadable asset matches, so rooms are never
    left empty. The local mesh is built at the footprint's real size, then
    grounded/scaled into place by the caller via normalize_furniture_to_box.
    """
    ymin, xmin, ymax, xmax = box
    w = max((xmax - xmin) * SCALE, 0.2)
    d = max((ymax - ymin) * SCALE, 0.2)

    wood = pbr_material((150, 110, 74, 255), "proc_wood")
    fab = pbr_material((120, 132, 150, 255), "proc_fabric")
    ceram = pbr_material((238, 240, 242, 255), "proc_ceramic")
    metal = pbr_material((200, 203, 208, 255), "proc_metal")

    cls = (furniture_class or "").lower()
    builder, mat = _proc_generic, wood
    if "bed" in cls:
        builder, mat = _proc_bed, fab
    elif any(k in cls for k in ("sofa", "couch", "sectional", "loveseat")):
        builder, mat = _proc_sofa, fab
    elif "dining" in cls or ("table" in cls and "bed" not in cls):
        builder, mat = _proc_table, wood
    elif "chair" in cls or "stool" in cls or "armchair" in cls:
        builder, mat = _proc_chair, wood
    elif "desk" in cls:
        builder, mat = _proc_desk, wood
    elif any(k in cls for k in ("wardrobe", "closet", "cabinet", "dresser", "bookshelf", "shelf")):
        builder, mat = _proc_wardrobe, wood
    elif "bath" in cls or "tub" in cls:
        builder, mat = _proc_bathtub, ceram
    elif "toilet" in cls or "wc" in cls:
        builder, mat = _proc_toilet, ceram
    elif "sink" in cls or "vanity" in cls or "basin" in cls:
        builder, mat = _proc_sink, ceram
    elif "stove" in cls or "oven" in cls or "cooktop" in cls or "range" in cls:
        builder, mat = _proc_stove, metal
    elif "fridge" in cls or "refrigerator" in cls or "freezer" in cls:
        builder, mat = _proc_fridge, metal

    return builder(w, d, mat, fab)


# ---------------------------------------------------------------------------
# Ceiling / roof
# ---------------------------------------------------------------------------

def build_ceiling() -> trimesh.Trimesh:
    """A thin slab over the whole footprint, sitting on top of the walls."""
    slab = scale_mesh([0, 0, 1000, 1000], WALL_HEIGHT, WALL_HEIGHT + 0.12)
    return mesh_with_material(slab, pbr_material((236, 235, 231, 255), "ceiling"))


# ---------------------------------------------------------------------------
# Room wall coloring
# ---------------------------------------------------------------------------

def _hex_to_rgba(value: Any) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(value, str):
        return None
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    # Pull toward a bright wall tone so colors read as paint, not neon.
    blend = 0.55
    r = int(r + (238 - r) * (1 - blend))
    g = int(g + (238 - g) * (1 - blend))
    b = int(b + (234 - b) * (1 - blend))
    return (r, g, b, 255)


def _room_wall_material(room_colors: Optional[dict[str, Any]]) -> trimesh.visual.material.PBRMaterial:
    """Pick a single accent wall color from Gemini's room_colors (average)."""
    if not room_colors:
        return pbr_material((238, 238, 234, 255), "wall_white")
    rgba = None
    for v in room_colors.values():
        rgba = _hex_to_rgba(v)
        if rgba:
            break
    if not rgba:
        return pbr_material((238, 238, 234, 255), "wall_white")
    return pbr_material(rgba, "wall_room")


# ---------------------------------------------------------------------------
# POST /generate-3d
# ---------------------------------------------------------------------------

def assemble_glb_bytes(
    elements: list[dict[str, Any]],
    furniture_assets: list[tuple[dict[str, Any], bytes]],
    include_ceiling: bool = False,
    room_colors: Optional[dict[str, Any]] = None,
) -> tuple[bytes, list[str]]:
    wall_mat = _room_wall_material(room_colors)
    scene, logs = build_shell(elements, wall_material=wall_mat)

    # Optional ceiling/roof. The node is named "ceiling" so the viewer can hide it.
    if include_ceiling:
        scene.add_geometry(build_ceiling(), node_name="ceiling", geom_name="ceiling")
        logs.append("Added ceiling/roof slab.")

    if room_colors:
        logs.append(f"Applied room wall tint from Gemini room_colors ({len(room_colors)} room(s)).")

    placed_real = 0
    placed_proc = 0
    fetched_boxes = {id(el) for el, _ in furniture_assets}
    real_nodes: list[str] = []

    for i, (el, data) in enumerate(furniture_assets):
        name = el.get("furniture_class") or f"furniture_{i}"
        try:
            loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False)
            mesh = _as_one_mesh(loaded)
            mesh = normalize_furniture_to_box(mesh, el["box_2d"])
            node = f"furn_{i}_{name}"
            scene.add_geometry(mesh, node_name=node, geom_name=node)
            real_nodes.append(node)
            placed_real += 1
        except Exception as exc:
            logs.append(f"Could not place asset for '{name}' ({type(exc).__name__}: {exc}) — will synthesize instead.")
            fetched_boxes.discard(id(el))

    # Procedural fallback: every furniture element that did NOT get a real asset
    # becomes a recognizable placeholder so no room is left empty.
    furniture_elements = [e for e in elements if e["element_type"] == "furniture"]
    for j, el in enumerate(furniture_elements):
        if id(el) in fetched_boxes:
            continue
        name = el.get("furniture_class") or "furniture"
        try:
            mesh = build_procedural_furniture(name, el["box_2d"])
            mesh = normalize_furniture_to_box(mesh, el["box_2d"])
            node = f"proc_{j}_{name}"
            scene.add_geometry(mesh, node_name=node, geom_name=node)
            placed_proc += 1
        except Exception as exc:
            logs.append(f"Could not synthesize placeholder for '{name}' ({type(exc).__name__}: {exc}) — skipping.")

    if furniture_elements:
        logs.append(
            f"Furniture: {placed_real} matched asset(s) from the web, "
            f"{placed_proc} procedural placeholder(s), {len(furniture_elements)} total."
        )

    # Stash a small manifest in the GLB's scene extras so the viewer can tell
    # ceiling/real/procedural nodes apart.
    try:
        scene.metadata["dolgen"] = {
            "ceiling_node": "ceiling" if include_ceiling else None,
            "real_furniture": real_nodes,
            "placed_real": placed_real,
            "placed_procedural": placed_proc,
        }
    except Exception:
        pass

    glb = scene.export(file_type="glb")
    logs.append(f"Exported dollhouse GLB ({len(glb)//1024} KB).")
    return glb, logs


def create_app() -> FastAPI:
    web = FastAPI(title="DolGen — Floorplan to 3D Dollhouse")

    # The React frontend (Vite dev server or a deployed static build) calls
    # /generate-3d cross-origin; the custom headers must be exposed for the
    # pipeline log / furniture source to be readable from JS.
    web.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-DolGen-Log", "X-DolGen-Furniture", "Content-Disposition"],
    )

    @web.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @web.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @web.post("/generate-3d")
    async def generate_3d(
        file: UploadFile = File(...),
        ceiling: str = Form("false"),
        room_colors: str = Form(""),
        furniture_source: str = Form("auto"),
    ) -> StreamingResponse:
        mime = (file.content_type or "").lower()
        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="Empty upload.")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=422, detail="Image is larger than 20 MB.")
        if not (mime.startswith("image/") or file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
            raise HTTPException(status_code=422, detail=f"Unsupported upload type '{file.content_type}'. Send an image (png/jpg/webp).")
        if not mime.startswith("image/"):
            mime = "image/png"

        furniture_source = (furniture_source or "auto").strip().lower()
        if furniture_source not in ("auto", "assets", "ai", "procedural"):
            raise HTTPException(
                status_code=422,
                detail=f"Unknown furniture_source '{furniture_source}' (auto|assets|ai|procedural).",
            )

        parsed = await extract_elements(data, mime)
        elements = normalize_elements(parsed)
        if not elements:
            raise HTTPException(status_code=422, detail="No usable elements were detected in the floorplan.")
        if not any(e["element_type"] == "wall" for e in elements):
            # Walls are the backbone of the dollhouse; without them the output is misleading.
            raise HTTPException(status_code=422, detail="Gemini detected no walls in this floorplan.")

        # Optional per-room wall colors suggested by Gemini (or sent by the client).
        colors: Optional[dict[str, Any]] = None
        if room_colors:
            try:
                obj = json.loads(room_colors)
                if isinstance(obj, dict):
                    colors = obj
            except (ValueError, TypeError):
                logger.info("Ignoring malformed room_colors form field")
        include_ceiling = str(ceiling).lower() in ("1", "true", "yes", "on")

        # Resolve furniture models per the requested source.
        asset_logs: list[str] = []
        if furniture_source == "procedural":
            furniture_assets: list[tuple[dict[str, Any], bytes]] = []
            mode_label = "procedural"
        elif furniture_source == "assets":
            furniture_assets, asset_logs = await download_assets(elements)
            mode_label = "assets"
        elif furniture_source == "ai":
            furniture_assets, asset_logs = await generate_ai_furniture_assets(elements, data)
            mode_label = "ai"
        else:  # auto: web asset library first, AI generation for the misses
            furniture_assets, asset_logs = await download_assets(elements)
            matched = {id(el) for el, _ in furniture_assets}
            remaining = [
                e
                for e in elements
                if e["element_type"] == "furniture" and id(e) not in matched
            ]
            if remaining:
                ai_assets, ai_logs = await generate_ai_furniture_assets(remaining, data)
                furniture_assets = furniture_assets + ai_assets
                asset_logs += ai_logs
            mode_label = "auto"

        glb, scene_logs = assemble_glb_bytes(
            elements, furniture_assets, include_ceiling=include_ceiling, room_colors=colors
        )

        logs = asset_logs + scene_logs
        logger.info("generate_3d ok: %s", " | ".join(logs))
        # HTTP headers are latin-1; keep the pipeline log ASCII-safe for the wire.
        header_log = " | ".join(logs).encode("ascii", "replace").decode("ascii")[:1800]
        return StreamingResponse(
            io.BytesIO(glb),
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": 'attachment; filename="dollhouse.glb"',
                "X-DolGen-Log": header_log,
                "X-DolGen-Furniture": mode_label,
            },
        )

    return web


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("gemini-secret"),
        modal.Secret.from_name("asset-api-secret"),
    ],
    timeout=600,
    memory=2048,
)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    """Modal entrypoint: builds and serves the FastAPI app with secrets injected."""
    return create_app()


# ---------------------------------------------------------------------------
# Frontend — Three.js + three-gpu-pathtracer (no <model-viewer>)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DolGen — Floorplan → 3D Dollhouse</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #16181c; color: #e8eaed; display: flex; height: 100vh; overflow: hidden;
  }
  #panel {
    width: 320px; min-width: 320px; padding: 20px; background: #1d2025;
    border-right: 1px solid #2c3037; display: flex; flex-direction: column; gap: 14px;
  }
  #panel h1 { font-size: 19px; margin: 0; letter-spacing: .2px; }
  #panel h1 span { color: #e8b04b; }
  #drop {
    border: 2px dashed #3a404c; border-radius: 10px; padding: 18px; text-align: center;
    cursor: pointer; transition: border-color .15s, background .15s; position: relative;
    background: #20242a; min-height: 150px; display: flex; align-items: center; justify-content: center;
  }
  #drop:hover, #drop.drag { border-color: #5f6e85; background: #23282f; }
  #drop input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  #drop .hint { font-size: 13px; color: #9aa3af; line-height: 1.5; pointer-events: none; }
  #drop img { max-width: 100%; max-height: 200px; border-radius: 6px; pointer-events: none; }
  #filename { font-size: 12px; color: #8f98a5; word-break: break-all; }
  #go {
    background: #e8b04b; border: none; color: #1c1710; font-weight: 700; font-size: 14px;
    padding: 11px 14px; border-radius: 8px; cursor: pointer; transition: filter .15s;
  }
  #go:disabled { filter: grayscale(.8) brightness(.6); cursor: not-allowed; }
  #go:not(:disabled):hover { filter: brightness(1.08); }
  #stats { font-size: 12px; color: #8f98a5; }
  #stats .converged { color: #55b785; font-weight: 600; }
  #log {
    flex: 1; overflow-y: auto; background: #16191d; border: 1px solid #292e36; border-radius: 8px;
    padding: 10px 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px; line-height: 1.55; color: #aeb6c2; white-space: pre-wrap;
  }
  #log .err { color: #ff7b72; }
  #log .ok { color: #55b785; }
  #view { position: relative; flex: 1; }
  #view canvas { display: block; width: 100%; height: 100%; }
  #empty {
    position: absolute; inset: 0; display: flex; flex-direction: column; gap: 10px;
    align-items: center; justify-content: center; color: #5c6572; pointer-events: none; text-align: center;
  }
  #empty .big { font-size: 44px; }
  #busy {
    position: fixed; inset: 0; background: rgba(10,12,15,.72); display: none; z-index: 10;
    align-items: center; justify-content: center; flex-direction: column; gap: 16px;
  }
  #busy.show { display: flex; }
  .dots span { animation: blink 1.2s infinite both; font-size: 26px; color: #e8b04b; }
  .dots span:nth-child(2) { animation-delay: .2s; }
  .dots span:nth-child(3) { animation-delay: .4s; }
  @keyframes blink { 0%,80%,100% { opacity:.15; } 40% { opacity:1; } }
  #busymsg { font-size: 13px; color: #c7ceda; }
  .row { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: #aeb6c2; flex-wrap: wrap; }
  .row label { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
  .row input[type="checkbox"] { accent-color: #e8b04b; width: 15px; height: 15px; cursor: pointer; }
  .row input[type="range"] { accent-color: #e8b04b; flex: 1; min-width: 90px; }
  .row .val { color: #e8b04b; font-variant-numeric: tabular-nums; min-width: 3.2em; text-align: right; }
  #dl {
    background: transparent; border: 1px solid #3a404c; color: #aeb6c2; font-size: 12.5px;
    padding: 8px 12px; border-radius: 8px; cursor: pointer; transition: border-color .15s, color .15s;
  }
  #dl:not(:disabled):hover { border-color: #e8b04b; color: #e8b04b; }
  #dl:disabled { opacity: .45; cursor: not-allowed; }
</style>
</head>
<body>
  <aside id="panel">
    <h1>Dol<span>Gen</span> 🏠</h1>
    <div id="drop">
      <input id="file" type="file" accept="image/png,image/jpeg,image/webp" />
      <div class="hint">Drop a floorplan image here<br/>or click to browse</div>
    </div>
    <div id="filename"></div>
    <button id="go" disabled>Generate Path-Traced Dollhouse</button>
    <div class="row">
      <label><input type="checkbox" id="ceil" /> Ceiling / roof</label>
      <label><input type="checkbox" id="spin" /> Auto-rotate</label>
    </div>
    <div class="row">
      <label for="quality">Quality</label>
      <input type="range" id="quality" min="128" max="4096" step="128" value="1024" />
      <span class="val" id="qualityVal">1024</span>
    </div>
    <button id="dl" disabled>⬇ Download .glb</button>
    <div id="stats"></div>
    <div id="log"></div>
  </aside>
  <main id="view">
    <div id="empty"><div class="big">📐 → 🏠</div><div>Upload a floorplan to render your dollhouse.<br/>Progressive path tracing will converge once the model loads.</div></div>
  </main>
  <div id="busy"><div class="dots"><span>●</span><span>●</span><span>●</span></div><div id="busymsg">Analyzing with Gemini — this takes ~10–30 s</div></div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.1/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.1/examples/jsm/",
    "three-mesh-bvh": "https://unpkg.com/three-mesh-bvh@0.7.8/build/index.module.js",
    "three-gpu-pathtracer": "https://unpkg.com/three-gpu-pathtracer@0.16.0/build/index.module.js"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { WebGLPathTracer } from 'three-gpu-pathtracer';

const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const goBtn = document.getElementById('go');
const logEl = document.getElementById('log');
const statsEl = document.getElementById('stats');
const busyEl = document.getElementById('busy');
const emptyEl = document.getElementById('empty');
const filenameEl = document.getElementById('filename');
const viewEl = document.getElementById('view');
const ceilEl = document.getElementById('ceil');
const spinEl = document.getElementById('spin');
const dlEl = document.getElementById('dl');
const qualityEl = document.getElementById('quality');
const qualityValEl = document.getElementById('qualityVal');

function log(msg, cls = '') {
  const div = document.createElement('div');
  if (cls) div.className = cls;
  div.textContent = msg;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

let pickedFile = null;

// ---- upload UI ------------------------------------------------------------
function pick(file) {
  if (!file) return;
  pickedFile = file;
  goBtn.disabled = false;
  filenameEl.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  const hint = drop.querySelector('.hint');
  if (hint) hint.remove();
  drop.insertBefore(img, drop.firstChild);
  log(`Selected ${file.name}`);
}
fileInput.addEventListener('change', () => pick(fileInput.files[0]));
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('drag');
  pick(e.dataTransfer.files[0]);
});

// ---- renderer & path tracer -----------------------------------------------
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.setPixelRatio(window.devicePixelRatio);
viewEl.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
camera.position.set(9, 11, 9);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.5, 0);
controls.autoRotateSpeed = 1.4;

const pathTracer = new WebGLPathTracer(renderer);
pathTracer.tiles = 3;
pathTracer.renderScale = 0.85;
pathTracer.dynamicLowRes = true;
pathTracer.targetSamples = parseInt(qualityEl.value, 10);
pathTracer.setCamera(camera);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1d22);

let emptyScene = true, convergedLogged = false, lastGlbBuf = null;
let ceilingMesh = null, ceilingOn = false;

pathTracer.setScene(scene, camera);

function retarget() {
  pathTracer.targetSamples = parseInt(qualityEl.value, 10);
  pathTracer.reset();
  convergedLogged = false;
}
qualityEl.addEventListener('input', () => { qualityValEl.textContent = qualityEl.value; });
qualityEl.addEventListener('change', retarget);

function setCeilingVisible(visible) {
  ceilingOn = visible;
  if (ceilingMesh) ceilingMesh.visible = visible;
  pathTracer.reset();
  convergedLogged = false;
}
ceilEl.addEventListener('change', () => setCeilingVisible(ceilEl.checked));

controls.addEventListener('change', () => {
  pathTracer.reset();
  convergedLogged = false;
});
controls.addEventListener('start', () => { pathTracer.visible = false; });
controls.addEventListener('end',   () => { pathTracer.visible = true; });

function resize() {
  const w = viewEl.clientWidth, h = viewEl.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  pathTracer.reset();
}
new ResizeObserver(resize).observe(viewEl);
resize();

renderer.setAnimationLoop(() => {
  controls.autoRotate = spinEl.checked;
  controls.update();
  if (emptyScene || camera.position.y <= 0.03) {
    pathTracer.reset();
    pathTracer.updateCamera();
    pathTracer.renderSample();
  }
  const n = Math.floor(pathTracer.samples);
  if (!emptyScene && !convergedLogged && n >= pathTracer.targetSamples) {
    log(`Path tracing converged at ${n} samples.`, 'ok');
    convergedLogged = true;
  }
  statsEl.innerHTML = !emptyScene
    ? `samples: ${n} / ${pathTracer.targetSamples}${n >= pathTracer.targetSamples ? ' <span class="converged">— converged</span>' : ''}`
    : '';
});

// ---- model loading ---------------------------------------------------------
function fitCameraToObject(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const d = Math.max(sphere.radius * 2.4, 4);
  camera.position.set(center.x + d * 0.75, d * 0.85, center.z + d * 0.75);
  controls.target.copy(center).setY(Math.min(center.y, 0.8));
  camera.updateProjectionMatrix();
}

async function generate() {
  if (!pickedFile) return;
  goBtn.disabled = true;
  busyEl.classList.add('show');
  log('Uploading floorplan to Gemini…');
  try {
    const form = new FormData();
    form.append('file', pickedFile);
    form.append('ceiling', ceilEl.checked ? 'true' : 'false');
    const res = await fetch('/generate-3d', { method: 'POST', body: form });
    if (!res.ok) {
      let detail = await res.text();
      try { detail = JSON.parse(detail).detail || detail; } catch {}
      throw new Error(`server ${res.status}: ${detail}`);
    }
    const serverLog = res.headers.get('X-DolGen-Log');
    if (serverLog) serverLog.split(' | ').forEach(l => log(l, l.includes('failed') ? 'err' : ''));

    const buf = await res.arrayBuffer();
    lastGlbBuf = buf;
    dlEl.disabled = false;
    log(`Received ${(buf.byteLength / 1024).toFixed(0)} KB GLB — loading…`);
    const gltf = await new GLTFLoader().parseAsync(buf, '');

    scene.clear();
    scene.add(gltf.scene);
    emptyScene = false;
    emptyEl.style.display = 'none';

    // If the model has a ceiling node, show/hide it per the checkbox.
    ceilingMesh = null;
    gltf.scene.traverse(o => { if (!ceilingMesh && o.name === 'ceiling') ceilingMesh = o; });
    setCeilingVisible(ceilEl.checked && !!ceilingMesh);
    if (ceilEl.checked && !ceilingMesh) log('No ceiling slab in this model (it was not generated).', 'err');

    fitCameraToObject(gltf.scene);

    pathTracer.updateCamera();
    pathTracer.setScene(gltf.scene, camera);
    pathTracer.reset();
    convergedLogged = false;
    log('Scene handed to path tracer — watch it converge.', 'ok');
  } catch (err) {
    log(`Generation failed: ${err.message}`, 'err');
    if (/fetch/i.test(err.message)) log('Network error — is the backend running?', 'err');
  } finally {
    busyEl.classList.remove('show');
    goBtn.disabled = false;
  }
}
goBtn.addEventListener('click', generate);

dlEl.addEventListener('click', () => {
  if (!lastGlbBuf) return;
  const blob = new Blob([lastGlbBuf], { type: 'model/gltf-binary' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'dollhouse.glb';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  log('Saved dollhouse.glb', 'ok');
});
</script>
</body>
</html>
"""
