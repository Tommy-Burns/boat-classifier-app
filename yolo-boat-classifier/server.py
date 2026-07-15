import os
import json
import uuid
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import rasterio
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.transform import xy
from pyproj import Transformer
from PIL import Image
import mercantile
import io

# dirs
BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR  = BASE_DIR / "models"

for d in (UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)

# app
app = FastAPI(title="Boat Detector", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

jobs: dict[str, dict] = {}

# Detection defaults. `confidence` is a fixed detection baseline:
# the model returns every detection at/above 0.10 and the frontend slider handles
# user-side filtering after detection.
DEFAULT_PARAMS = {
    "confidence": 0.10,
    "slice_size": 640,
    "overlap":    0.2,
    "classes":    ["boat", "surfboard"],
    "model_size": "m",
}

# Classes offered by the UI class picker. The model can detect all COCO-80 classes,
# but this app intentionally limits targets to boats and surfboards.
AVAILABLE_CLASSES = ["boat", "surfboard"]

# YOLO26-seg model sizes. The weight file for a chosen size need not be present in
# MODEL_DIR — ultralytics auto-downloads it to that path on first load.
ALLOWED_MODEL_SIZES = ("n", "s", "m", "l", "x")


# helpers

def to_python(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def safe_json(data: dict) -> Response:
    return Response(content=json.dumps(to_python(data)), media_type="application/json")


def get_cog_meta(path: str) -> dict:
    """Read map bounds (WGS84) + raster geometry from the COG."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("COG has no CRS — it must be georeferenced to be used as a map basemap.")
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return {
            "bounds":  [float(w), float(s), float(e), float(n)],
            "center":  [float((w + e) / 2), float((s + n) / 2)],
            "crs":     str(src.crs),
            "width":   src.width,
            "height":  src.height,
        }


def get_image_size(path: str) -> tuple[int, int]:
    """Return (width, height) of the inference image."""
    with Image.open(path) as im:
        return im.width, im.height


# detection

def run_detection(inference_path: str, cog_path: str, job_id: str, params: dict | None = None):
    """
    Run boat detection on the inference IMAGE, then georeference the resulting
    pixel coordinates through the COG's geotransform so the polygons land in the
    correct geographic position on the map.

    ASSUMPTION: the inference image and the COG cover the SAME geographic extent
    (the image is a rendering/export of the same scene as the COG). Pixel
    coordinates are scaled by the width/height ratio between the two rasters,
    then mapped to geographic coordinates via the COG transform, and finally
    reprojected to EPSG:4326 for Leaflet.

    `params` (all optional, defaults preserve the original behaviour displayed on the frontend — see DEFAULT_PARAMS):
      * confidence – detection baseline. We run at a low baseline and return EVERY
        detection at/above it so the frontend can re-filter client-side with a
        slider without re-running the model.
      * slice_size – SAHI slice height/width in px.
      * overlap    – SAHI overlap ratio (both axes).
      * classes    – list of target class names to keep (e.g. ["boat"]).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    confidence = float(params["confidence"])
    slice_size = int(params["slice_size"])
    overlap    = float(params["overlap"])
    classes    = [c.strip() for c in params["classes"] if str(c).strip()] or ["boat"]
    class_set  = set(classes)
    model_size = params["model_size"] if params["model_size"] in ALLOWED_MODEL_SIZES else "m"

    try:
        jobs[job_id]["status"]   = "running"
        jobs[job_id]["progress"] = "Loading model…"
        jobs[job_id]["params"]   = {
            "confidence": confidence, "slice_size": slice_size,
            "overlap": overlap, "classes": classes, "model_size": model_size,
        }

        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.slicing import get_slice_bboxes

        # ultralytics auto-downloads this to MODEL_DIR on first use if absent
        model_path = MODEL_DIR / f"yolo26{model_size}-seg.pt"

        # device fallback
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=confidence,
            device=device,
        )

        # total slice count up front so progress is a real "x / N"
        inf_w, inf_h = get_image_size(inference_path)
        try:
            total_slices = len(get_slice_bboxes(
                image_height=inf_h, image_width=inf_w,
                slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            ))
        except Exception:
            total_slices = 0

        jobs[job_id]["slice_total"]   = total_slices
        jobs[job_id]["slice_current"] = 0
        jobs[job_id]["progress"]      = (
            f"Processing slice 0 / {total_slices}" if total_slices
            else "Running sliced inference…"
        )

        def on_progress(done, total):
            jobs[job_id]["slice_current"] = int(done)
            jobs[job_id]["slice_total"]   = int(total)
            jobs[job_id]["progress"]      = f"Processing slice {int(done)} / {int(total)}"

        result = get_sliced_prediction(
            inference_path,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMM",
            postprocess_match_threshold=0.5,
            progress_callback=on_progress,
        )

        boat_predictions = [
            p for p in result.object_prediction_list
            if p.category.name in class_set
        ]

        jobs[job_id]["progress"] = "Georeferencing detections…"

        # inference image dimensions (inf_w/inf_h already computed above for slicing)
        # COG georeferencing context
        with rasterio.open(cog_path) as cog:
            cog_transform = cog.transform
            cog_crs       = cog.crs
            cog_w, cog_h  = cog.width, cog.height

        # transformer: COG CRS - WGS84 (lon/lat) for Leaflet
        to_wgs84 = Transformer.from_crs(cog_crs, "EPSG:4326", always_xy=True)

        # scale factors from inference-pixel space - COG-pixel space
        sx = cog_w / inf_w
        sy = cog_h / inf_h

        def pixel_to_lonlat(px, py):
            cog_col = px * sx
            cog_row = py * sy
            mx, my  = xy(cog_transform, cog_row, cog_col)   # COG CRS units
            lon, lat = to_wgs84.transform(mx, my)           # - WGS84
            return [float(lon), float(lat)]

        features = []
        for pred in boat_predictions:
            if pred.mask is None:
                continue
            mask = pred.mask.bool_mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                if len(contour) < 3:
                    continue
                points = contour.squeeze()
                if len(points.shape) == 1:
                    continue

                coords = [pixel_to_lonlat(float(p[0]), float(p[1])) for p in points]
                coords.append(coords[0])  # close polygon

                bbox = pred.bbox
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "id":             str(uuid.uuid4())[:8],
                        "class":          pred.category.name,
                        "confidence":     float(pred.score.value),
                        "confidence_pct": f"{pred.score.value * 100:.1f}%",
                    },
                })

        geojson      = {"type": "FeatureCollection", "features": features}
        geojson_path = OUTPUT_DIR / f"{job_id}_boats.geojson"
        with open(geojson_path, "w") as f:
            json.dump(geojson, f)

        jobs[job_id].update({
            "status":      "done",
            "progress":    "Complete",
            "boat_count":  len(features),
            "geojson_url": f"/output/{job_id}_boats.geojson",
        })

    except Exception as e:
        import traceback
        jobs[job_id].update({
            "status":   "error",
            "progress": "Failed",
            "error":    str(e),
            "trace":    traceback.format_exc(),
        })


# tile server

def render_tile(cog_path: str, z: int, x: int, y: int) -> bytes:
    tile   = mercantile.Tile(x, y, z)
    bounds = mercantile.bounds(tile)

    with rasterio.open(cog_path) as src:
        t = Transformer.from_crs("EPSG:4326", str(src.crs), always_xy=True)
        left,  bottom = t.transform(bounds.west, bounds.south)
        right, top    = t.transform(bounds.east, bounds.north)

        tile_size = 256
        window    = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
        bands     = min(src.count, 3)
        data      = src.read(
            list(range(1, bands + 1)),
            window=window,
            out_shape=(bands, tile_size, tile_size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )

    if data.dtype != np.uint8:
        data = data.astype(np.float32)
        for i in range(data.shape[0]):
            band  = data[i]
            valid = band[band > 0]
            lo, hi = (np.percentile(valid, (2, 98)) if valid.size else (0, 1))
            data[i] = np.clip((band - lo) / max(hi - lo, 1e-9) * 255, 0, 255)
        data = data.astype(np.uint8)

    if bands == 1:
        data = np.repeat(data, 3, axis=0)

    # build an alpha channel so nodata areas are transparent
    rgb   = np.transpose(data, (1, 2, 0))
    alpha = (np.any(rgb > 0, axis=2) * 255).astype(np.uint8)
    rgba  = np.dstack([rgb, alpha])

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def blank_tile() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


# routes

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/classes")
def list_classes():
    """Class names offered by the UI class picker (boats + surfboards only)."""
    return safe_json({"classes": AVAILABLE_CLASSES, "default": DEFAULT_PARAMS["classes"]})


@app.post("/api/detect")
async def detect(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Image used for inference"),
    cog:   UploadFile = File(..., description="Georeferenced COG used for the map basemap"),
    slice_size: int   | None = Form(None, description="SAHI slice size in px"),
    overlap:    float | None = Form(None, description="SAHI overlap ratio (0–1)"),
    classes:    str   | None = Form(None, description="Comma-separated target class names"),
    model_size: str   | None = Form(None, description="YOLO26-seg size: n, s, m, l, or x"),
):
    """
    Both files are required:
      * image - run detection on this
      * cog   - georeferenced basemap; also used to place detections on the map

    The detection baseline (confidence) is fixed at DEFAULT_PARAMS["confidence"] — the
    frontend confidence slider does user-side filtering after detection.
    """
    job_id = str(uuid.uuid4())[:8]

    # build params, falling back to DEFAULT_PARAMS for anything omitted
    params = dict(DEFAULT_PARAMS)
    if slice_size is not None:
        params["slice_size"] = max(64, int(slice_size))
    if overlap is not None:
        params["overlap"] = max(0.0, min(float(overlap), 0.9))
    if classes is not None:
        picked = [c.strip() for c in classes.split(",")
                  if c.strip() in AVAILABLE_CLASSES]
        params["classes"] = picked or DEFAULT_PARAMS["classes"]
    if model_size is not None:
        size = model_size.strip().lower()
        if size not in ALLOWED_MODEL_SIZES:
            raise HTTPException(
                400, f"model_size must be one of {', '.join(ALLOWED_MODEL_SIZES)}")
        params["model_size"] = size

    img_ext = Path(image.filename).suffix.lower()
    cog_ext = Path(cog.filename).suffix.lower()

    if cog_ext not in (".tif", ".tiff"):
        raise HTTPException(400, "COG must be a .tif / .tiff file")

    inf_path = UPLOAD_DIR / f"{job_id}_inf{img_ext}"
    cog_path = UPLOAD_DIR / f"{job_id}_cog{cog_ext}"

    try:
        with open(inf_path, "wb") as f:
            f.write(await image.read())
        with open(cog_path, "wb") as f:
            f.write(await cog.read())

        # validate the COG is georeferenced before starting a job
        meta = get_cog_meta(str(cog_path))

        job = {
            "job_id":         job_id,
            "status":         "queued",
            "progress":       "Queued",
            "inference_path": str(inf_path),
            "cog_path":       str(cog_path),
            "image_name":     image.filename,
            "cog_name":       cog.filename,
            "meta":           meta,
            "tile_url":       f"/tiles/{job_id}/{{z}}/{{x}}/{{y}}.png",
            "params":         params,
            "created_at":     datetime.utcnow().isoformat(),
        }
        jobs[job_id] = job
        background_tasks.add_task(run_detection, str(inf_path), str(cog_path), job_id, params)
        return safe_json(job)

    except HTTPException:
        raise
    except Exception as e:
        inf_path.unlink(missing_ok=True)
        cog_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Upload failed: {e}")


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return safe_json(jobs[job_id])


@app.get("/tiles/{job_id}/{z}/{x}/{y}.png")
def tile(job_id: str, z: int, x: int, y: int):
    if job_id not in jobs:
        return Response(content=blank_tile(), media_type="image/png")
    try:
        png = render_tile(jobs[job_id]["cog_path"], z, x, y)
        return Response(content=png, media_type="image/png")
    except Exception:
        return Response(content=blank_tile(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
