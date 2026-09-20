"""
DolGen — turn a 2D floorplan image into a path-traced 3D dollhouse.

Structure only: the pipeline reconstructs the architectural shell of the house —
walls, door/window openings, floor (and optional ceiling). Furniture is
deliberately discarded so the extraction model can spend all of its attention
on the structure.

Pipeline (per POST /generate-3d request):
  1. OpenCV preprocessing normalizes the upload: EXIF orientation, upscale to
     ~2048 px, denoise, and white-balance/normalize the background (gray-world
     white balance + CLAHE contrast). The result is a binary wall mask: thick
     dark partitions (walls) become black lines on white, and small dark
     components inside rooms (hatching, furniture strokes, symbols) are
     removed — so the vision model sees clean structure only. On any failure
     the original bytes are used instead.
  2. Kimi K3 (Moonshot AI vision model, OpenAI-compatible endpoint on Modal,
     temperature=0.0, JSON-mode) extracts walls / doors / windows as 2D boxes
     normalized to a 0-1000 image coordinate system.
  3. Trimesh assembles the dollhouse: 2.5 m walls, boolean-cut door/window
     openings, PBR materials (white walls, light-wood floor).
  4. The merged scene is exported as binary GLTF (.glb) and streamed back with
     media type 'model/gltf-binary'.

GET / serves a legacy Three.js viewer (zero-build fallback). The primary
frontend is the React app in frontend/ (Vite) which renders the GLB with
three-gpu-pathtracer (progressive path tracing + ACES filmic tone mapping).

Modal setup (secrets are intentionally NOT baked into this file):
  modal secret create kimi-verify KIMI_BASE_URL=...   # plus KIMI_TOKEN_ID/KIMI_TOKEN_SECRET,
                                                      # or rely on the workspace MODAL_TOKEN_*
  modal serve app.py      # hot-reload playground URL
  modal deploy app.py     # persistent deployment
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Literal, Optional

import modal
import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator

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
        "openai",  # Kimi K3 is served over an OpenAI-compatible endpoint
        # trimesh[all] covers the glTF stack; manifold3d lives in trimesh's *easy*
        # extra, not [all], so it must be pinned explicitly for boolean operations.
        "trimesh[all]",
        "manifold3d>=2.3.0",
        "scipy",
        "networkx",
        "pillow",  # decode uploads before OpenCV preprocessing
        "opencv-python-headless",  # floorplan preprocessing (normalization + wall mask)
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

# Kimi K3 (Moonshot AI), self-hosted on Modal behind an OpenAI-compatible
# endpoint. There is no public URL — KIMI_BASE_URL must point at YOUR deployed
# endpoint (e.g. https://<workspace>--kimi-k3-serve.modal.run/v1). The endpoint
# authenticates with a token id + token secret pair.
#
# Credentials come from the "kimi-verify" Modal secret, which supplies
# KIMI_BASE_URL (and optionally KIMI_MODEL / KIMI_TIMEOUT_S). The token pair is
# read from KIMI_TOKEN_ID / KIMI_TOKEN_SECRET if present, falling back to the
# Modal workspace tokens MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (what a Modal-
# hosted endpoint typically expects).
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-k3")
KIMI_BASE_URL = os.environ.get("KIMI_BASE_URL", "")
KIMI_TIMEOUT_S = float(os.environ.get("KIMI_TIMEOUT_S", "120"))
MAX_IMAGE_BYTES = 20 * 1024 * 1024          # 20 MB upload cap

# Cross-origin access for the React dev server / any deployed static frontend.
# The Vite dev server proxies /generate-3d, so this is mainly for direct calls.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

# ---------------------------------------------------------------------------
# Floorplan preprocessing (OpenCV)
#
# The vision model is far more accurate on a clean structural drawing than on a raw
# scan/photo of a colored floorplan. Before extraction we:
#   1. honor EXIF orientation and upscale small images to ~2048 px,
#   2. denoise and normalize the background (gray-world white balance +
#      CLAHE contrast), so colored scans and photos behave like white plans,
#   3. build a binary *wall mask*: adaptive threshold -> morphology ->
#      small-connected-component removal. Thick dark partitions survive as
#      black lines on white; furniture strokes, hatching, and symbols inside
#      rooms (small dark components) are discarded.
# Any failure anywhere falls back to the raw upload bytes.
# ---------------------------------------------------------------------------

# Minimum surviving dark-component size, as a fraction of the image area.
# Real wall partitions are long thick runs (~0.05%+ of the image); furniture
# strokes / symbols / hatching are far smaller.
_WALL_MASK_MIN_AREA_FRAC = 5e-4


def _bytes_to_bgr(image_bytes: bytes) -> Optional[Any]:
    """Decode image bytes to a BGR array, honoring EXIF orientation."""
    import cv2
    from PIL import Image, ImageOps

    try:  # PIL first: applies EXIF rotation that cv2.imdecode ignores
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _normalize_floorplan(bgr: Any, long_side: int = 2048) -> Any:
    """Denoise + white-balance + contrast-normalize a floorplan scan/photo."""
    import cv2

    h, w = bgr.shape[:2]
    scale = long_side / max(h, w)
    if scale > 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Gray-world white balance: pull colored casts (sepia scans, tinted fills)
    # toward neutral before the heavy normalization.
    means = bgr.reshape(-1, 3).mean(axis=0)
    bgr = np.clip(bgr.astype(np.float32) * (means.mean() / np.maximum(means, 1e-6)), 0, 255).astype(np.uint8)

    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def build_wall_mask(gray: Any) -> Any:
    """Binary wall mask: walls black (0) on white (255); interior clutter removed.

    Otsu on an aggressively blurred image finds the ink/background split even on
    colored plans; an opening pass drops thin strokes, then small dark
    components (furniture, symbols, hatching) are discarded so only the thick
    structural partitions remain.
    """
    import cv2

    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(2, int(round(min(h, w) / 400)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(24, int(_WALL_MASK_MIN_AREA_FRAC * h * w))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    walls = np.zeros_like(ink)
    for i in range(1, n):  # label 0 is the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            walls[_labels == i] = 255
    return cv2.bitwise_not(walls)  # walls black on white


def preprocess_floorplan(image_bytes: bytes) -> tuple[bytes, str]:
    """Normalize the upload and return (png_bytes, 'image/png').

    Falls back to the original bytes and mime type when decoding fails.
    """
    bgr = _bytes_to_bgr(image_bytes)
    if bgr is None:
        return image_bytes, ""
    import cv2

    normalized = _normalize_floorplan(bgr)
    mask = build_wall_mask(normalized)
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        return image_bytes, ""
    return buf.tobytes(), "image/png"


# ---------------------------------------------------------------------------
# Kimi extraction schema
# ---------------------------------------------------------------------------

class FloorElement(BaseModel):
    """One extracted structural element. box_2d is [ymin, xmin, ymax, xmax]
    on a 0-1000 normalized image coordinate system ((0,0) = top-left)."""

    element_type: Literal["wall", "door", "window"]
    box_2d: list[float]                              # kept float here; rounded/clamped downstream

    @field_validator("box_2d")
    @classmethod
    def _box_has_four_sides(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("box_2d must have exactly 4 numbers [ymin, xmin, ymax, xmax]")
        return v


KIMI_PROMPT = """You are an architectural floorplan understanding engine.

The image is a preprocessed binary wall mask of a floorplan: structural walls are thick BLACK
lines on a WHITE background. Furniture and interior clutter have already been removed.

Extract every STRUCTURAL element of the house — walls, doors, windows — and return ONLY a
JSON array matching the provided schema. Focus exclusively on the structure of the building.

Rules:
- box_2d = [ymin, xmin, ymax, xmax] as integers normalized to 0-1000, where (0,0) is the
  TOP-LEFT of the image. Boxes must be axis-aligned and tight around the element.
- element_type "wall": every solid wall segment (exterior and interior). Represent each straight
  wall segment as its own thin box. Walls should join to form closed rooms where possible.
- element_type "door": every door opening through a wall (a gap in the black wall stroke,
  possibly with a swing arc). The box covers the opening rectangle within the wall.
- element_type "window": every window opening through a wall (a break in an exterior wall,
  often marked by parallel thin lines). The box covers the opening rectangle within the wall.
- The outer boundary of the black walls is the building footprint — trace it exactly.
- Do NOT invent rooms, floors, or walls that are not visible. Do NOT report furniture,
  fixtures, text labels, dimension lines, arrows, scale bars or north symbols.
- If a wall segment is ambiguous, prefer continuity: walls that clearly meet should be
  reported as segments that join.

Be thorough and precise: missing or misplaced walls make the 3D model wrong."""


def _kimi_client() -> AsyncOpenAI:
    """Build an OpenAI-compatible client for the Kimi K3 endpoint on Modal.

    The token pair is KIMI_TOKEN_ID / KIMI_TOKEN_SECRET when set, otherwise the
    Modal workspace tokens MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (what a Modal-
    hosted endpoint typically expects). KIMI_BASE_URL points at the endpoint.
    """
    token_id = os.environ.get("KIMI_TOKEN_ID") or os.environ.get("MODAL_TOKEN_ID", "")
    token_secret = os.environ.get("KIMI_TOKEN_SECRET") or os.environ.get("MODAL_TOKEN_SECRET", "")
    if not (KIMI_BASE_URL and token_id and token_secret):
        raise HTTPException(
            status_code=500,
            detail=(
                "Kimi endpoint is not configured — the 'kimi-verify' Modal secret must "
                "set KIMI_BASE_URL, plus KIMI_TOKEN_ID/KIMI_TOKEN_SECRET (or rely on the "
                "MODAL_TOKEN_ID/MODAL_TOKEN_SECRET workspace tokens)."
            ),
        )
    return AsyncOpenAI(
        base_url=KIMI_BASE_URL,
        api_key=f"{token_id}:{token_secret}",
        timeout=KIMI_TIMEOUT_S,
        max_retries=1,
    )


def _elements_json_schema() -> dict[str, Any]:
    """JSON schema for the extraction response (a raw array of FloorElement)."""
    schema = FloorElement.model_json_schema()
    schema.pop("title", None)
    return {
        "name": "floorplan_elements",
        "strict": True,
        "schema": {
            "type": "array",
            "items": schema,
        },
    }


def _parse_elements_text(text: Optional[str]) -> list[FloorElement]:
    """Parse the model's JSON text into validated FloorElements.

    Handles a raw array, an object wrapping the array, and ```json fences.
    """
    if not text:
        return []
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()
    try:
        obj = json.loads(s)
    except ValueError:
        return []
    if isinstance(obj, dict):  # tolerate a wrapper like {"elements": [...]}
        obj = next((v for v in obj.values() if isinstance(v, list)), [])
    if not isinstance(obj, list):
        return []
    out: list[FloorElement] = []
    for item in obj:
        try:
            out.append(FloorElement.model_validate(item))
        except Exception:
            continue
    return out


async def extract_elements(image_bytes: bytes, mime_type: str) -> list[FloorElement]:
    """Call Kimi K3 on the (preprocessed) floorplan; return validated elements."""
    client = _kimi_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    kwargs: dict[str, Any] = {
        "model": KIMI_MODEL,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                    {"type": "text", "text": KIMI_PROMPT},
                ],
            }
        ],
    }
    # JSON mode: prefer strict schema; fall back to plain JSON-object mode, then
    # to no constraint, so the call works across OpenAI-compatible servers.
    try:
        response = await client.chat.completions.create(
            **kwargs, response_format={"type": "json_schema", "json_schema": _elements_json_schema()}
        )
    except Exception:
        try:
            response = await client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
        except Exception:
            try:
                response = await client.chat.completions.create(**kwargs)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Kimi analysis failed: {exc}")

    text = response.choices[0].message.content if response.choices else None
    parsed = _parse_elements_text(text)
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
# (z_min, z_max) — walls, floor, and door/window cutter blocks
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

        Walls returned by the vision model are often fatter than real walls (it outlines the
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
        """Build a cutter for a door/window that is guaranteed to span the wall's
        thin axis (so the boolean produces a real through-hole even if the 2D boxes
        of the wall and opening don't perfectly overlap)."""
        oymin, oxmin, oymax, oxmax = op["box_2d"]
        wall_ext = wall_mesh.extents          # meters
        thin_axis = 0 if wall_ext[0] < wall_ext[1] else 1
        long_axis = 1 - thin_axis
        z_min, z_max = opening_hparams(op["box_2d"], kind)

        # Hole width: the opening's real extent along the wall's LONG axis, in meters.
        if long_axis == 0:
            op_long_lo, op_long_hi = oxmin * SCALE, oxmax * SCALE
        else:
            op_long_lo, op_long_hi = -oymax * SCALE, -oymin * SCALE
        if op_long_hi - op_long_lo <= 1e-3:
            return None

        # Cutter thin span = wall thin span + margin, so it passes fully through.
        margin = 0.05
        lo = [0.0, 0.0, z_min]
        hi = [0.0, 0.0, z_max]
        lo[long_axis], hi[long_axis] = op_long_lo, op_long_hi
        lo[thin_axis] = wall_mesh.bounds[0, thin_axis] - margin
        hi[thin_axis] = wall_mesh.bounds[1, thin_axis] + margin
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
    """Pick a single accent wall color from the model's room_colors (average)."""
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
        logs.append(f"Applied room wall tint from room_colors ({len(room_colors)} room(s)).")

    # Stash a small manifest in the GLB's scene extras so the viewer can find
    # the ceiling node.
    try:
        scene.metadata["dolgen"] = {
            "ceiling_node": "ceiling" if include_ceiling else None,
        }
    except Exception:
        pass

    glb = scene.export(file_type="glb")
    logs.append(f"Exported dollhouse GLB ({len(glb)//1024} KB).")
    return glb, logs


def create_app() -> FastAPI:
    web = FastAPI(title="DolGen — Floorplan to 3D Dollhouse (structure)")

    # The React frontend (Vite dev server or a deployed static build) calls
    # /generate-3d cross-origin; the custom headers must be exposed for the
    # pipeline log to be readable from JS.
    web.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-DolGen-Log", "Content-Disposition"],
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

        # Preprocess: EXIF fix, upscale, denoise, background normalization, and a
        # binary wall mask (structure only — furniture strokes are discarded).
        # Any failure keeps the raw upload so extraction still works.
        prep_logs: list[str] = []
        try:
            prepped, prepped_mime = preprocess_floorplan(data)
            if prepped_mime:
                data, mime = prepped, prepped_mime
                prep_logs.append("Preprocessed floorplan (denoise + white balance + wall mask).")
            else:
                prep_logs.append("Preprocessing could not decode the upload — using the raw image.")
        except Exception as exc:
            logger.warning("Preprocessing failed (%s); using raw upload", exc)
            prep_logs.append("Preprocessing failed — using the raw image.")

        parsed = await extract_elements(data, mime)
        elements = normalize_elements(parsed)
        if not elements:
            raise HTTPException(status_code=422, detail="No usable elements were detected in the floorplan.")
        if not any(e["element_type"] == "wall" for e in elements):
            # Walls are the backbone of the dollhouse; without them the output is misleading.
            raise HTTPException(status_code=422, detail="Kimi detected no walls in this floorplan.")

        # Optional per-room wall colors sent by the client.
        colors: Optional[dict[str, Any]] = None
        if room_colors:
            try:
                obj = json.loads(room_colors)
                if isinstance(obj, dict):
                    colors = obj
            except (ValueError, TypeError):
                logger.info("Ignoring malformed room_colors form field")
        include_ceiling = str(ceiling).lower() in ("1", "true", "yes", "on")

        glb, scene_logs = assemble_glb_bytes(
            elements, include_ceiling=include_ceiling, room_colors=colors
        )

        logs = prep_logs + scene_logs
        logger.info("generate_3d ok: %s", " | ".join(logs))
        # HTTP headers are latin-1; keep the pipeline log ASCII-safe for the wire.
        header_log = " | ".join(logs).encode("ascii", "replace").decode("ascii")[:1800]
        return StreamingResponse(
            io.BytesIO(glb),
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": 'attachment; filename="dollhouse.glb"',
                "X-DolGen-Log": header_log,
            },
        )

    return web


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("kimi-verify"),
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
  <div id="busy"><div class="dots"><span>●</span><span>●</span><span>●</span></div><div id="busymsg">Analyzing with Kimi K3 — this takes ~10–30 s</div></div>

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
  log('Uploading floorplan to Kimi K3…');
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
