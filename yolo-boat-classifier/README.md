# Boat Detection App

Detect boats in high-resolution aerial / satellite imagery and view them as a
georeferenced overlay on an interactive map.

The app runs **SAHI sliced inference** over a ** segmentation** model,
converts the detected boat masks into polygons, georeferences them through a
**Cloud-Optimised GeoTIFF (COG)**, and serves the result as a **GeoJSON overlay
on a Leaflet map**. The same COG is streamed to the map as XYZ tiles by a small
built-in tile server (no TiTiler / external tile service required).

---

## How it works

1. You upload two files: an **inference image** (what the model looks at) and a
   matching **georeferenced COG** (the map basemap).
2. The backend slices the inference image into overlapping tiles, runs the YOLO
   segmentation model on each slice, and merges the results.
3. Each boat mask is traced to a polygon, its pixel coordinates are mapped through
   the COG's geotransform, and the polygons are reprojected to WGS84 (EPSG:4326)
   so Leaflet can draw them in the right place.
4. The frontend renders the COG as map tiles, overlays the boat polygons, and
   gives you tools to filter, inspect, and export them.

### Why sliced inference (SAHI)?

Aerial / satellite scenes are huge - often tens of thousands of pixels per side.
Feeding a whole scene to the model at once either **runs out of GPU memory** or
forces such aggressive downscaling that **small boats disappear**. SAHI
(Slicing Aided Hyper Inference) cuts the image into manageable, overlapping
tiles (default 640×640), runs detection on each at full resolution, and stitches
the predictions back together with non-max merging. This keeps memory bounded
**and** preserves the detail needed to catch small vessels near the resolution
limit.

---

## Features

- **Configurable detection** - slice size, overlap ratio, and target classes
  (`boat` / `surfboard`) are exposed in a collapsible *Detection Settings* panel.
- **Real per-slice progress** - the status reads e.g. `Processing slice 47 / 180`
  while inference runs, backed by a real progress bar.
- **Post-detection confidence slider** - detection runs once at a low floor and
  returns *every* detection with its score; the slider re-filters the displayed
  polygons and the boat list **client-side**, with no model re-run. Boat count and
  average confidence update live.
- **Colour-by-confidence** - polygons are shaded on a red - yellow - green gradient,
  with a legend on the map.
- **Boat list** - every detection is listed in the side panel. Click a row to zoom
  to that boat and open its popup; hovering a row highlights its polygon and vice
  versa (two-way hover sync). The list is sortable by confidence and respects the
  confidence slider.
- **Layer controls** - overlay opacity slider and a show/hide toggle so the imagery
  underneath stays visible.
- **GeoJSON export** - download the full FeatureCollection of detections.

---

## Tech stack

| Layer | Tools |
| --- | --- |
| Backend / API | **FastAPI** + Uvicorn (run directly from `server.py`) |
| Detection | **Ultralytics YOLO26m-seg** wrapped by **SAHI** sliced inference |
| Mask - polygon | **OpenCV** (contour extraction), **NumPy** |
| Georeferencing & tiles | **rasterio** (windowed reads / transforms), **pyproj** (CRS reprojection), **mercantile** (XYZ tiling), **Pillow** (PNG encoding) |
| Frontend | **Leaflet 1.9** single-page UI (no build step), CartoDB dark basemap |

---

## Setup & installation

Recommended **Python 3.10+** or higher.

```bash
# from the project root
python3 -m venv .venv            # Windows: python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Model weights

Detection uses `yolo26m-seg.pt`, expected at `models/yolo26m-seg.pt`. If the file
isn't present, Ultralytics will **auto-download it on first run**. To place it
manually, drop the `yolo26m-seg.pt` weights into the `models/` directory.

> 💡 A CUDA GPU is used automatically if available; otherwise the app falls back to
> CPU (slower, but it works).

---

## Model Sizes

The app lets you choose between five YOLO segmentation model sizes via the **Model Size** dropdown. They share the same architecture but differ in the number of parameters, which trades inference speed against detection accuracy. Larger models detect more boats (especially small or partially occluded ones) but run slower and use more GPU memory.

| Size | Code | Relative speed | Relative accuracy | Best for |
|------|------|----------------|-------------------|----------|
| Nano | `n` | Fastest | Lowest | Quick tests, low-resource machines, large batches |
| Small | `s` | Fast | Lower | A step up from nano with modest cost |
| Medium | `m` | Balanced | Good | **Default** — the recommended balance of speed and quality |
| Large | `l` | Slow | High | When accuracy matters more than speed |
| Extra Large | `x` | Slowest | Highest | Maximum accuracy on difficult, dense scenes |


### Choosing a size

Since the app runs sliced inference (each slice is processed independently), the model size affects every slice, so the speed difference is amplified on large, high-resolution images. A practical approach:

- Start with **Medium (m)** - it handles most marina and harbour scenes well.
- If detections are slow and the scene is simple, drop to **Small (s)** or **Nano (n)**.
- If small boats are being missed in dense clusters, move up to **Large (l)** or **Extra Large (x)**, accepting the longer runtime.

Because the confidence slider re-filters results after detection without re-running the model, you can run once at a larger size and explore the results freely rather than re-running for each threshold.

---


## Running the app

```bash
python server.py
```

This starts Uvicorn on `http://0.0.0.0:8000`. Then open:

```
http://localhost:8000
```

> 💡 Run it with `python server.py` - **not** `uvicorn server:app`. The launch
> configuration lives in the `if __name__ == "__main__"` block at the bottom of
> `server.py`.

---

## Try it out

A ready-to-use sample pair is included in the **`test/`** folder so you can see the
app working immediately - no data prep needed:

- `test/boat_image.tif` - upload this as the **Inference Image**
- `test/boat_image_cog.tif` - upload this as the **Map COG**

Steps:

1. Start the app and open `http://localhost:8000`.
2. Drop `test/boat_image.tif` into the **Inference Image** zone.
3. Drop `test/boat_image_cog.tif` into the **Map COG** zone.
4. Click **Run Detection** and watch the per-slice progress.
5. When it finishes, the imagery loads as map tiles with the detected boats drawn
   on top. Use the confidence slider, boat list, and opacity controls to explore,
   then **Download GeoJSON** if you want the results.

---

## Preparing your own COG

The map basemap must be a georeferenced Cloud-Optimised GeoTIFF. Convert a regular
GeoTIFF with GDAL:

```bash
gdal_translate input.tif output_cog.tif -of COG
```

> ⚠️ **Do not add `TILING_SCHEME=GoogleMapsCompatible`.**
> That option reprojects and **pads the raster's extent out to the Web-Mercator
> tile grid**, which shifts the geotransform and **breaks the pixel-to-geo
> alignment** this app relies on to place detections correctly. It's also
> unnecessary here - the built-in tile server reprojects the COG to Web Mercator
> *on the fly* for each tile request. Keep the COG in its native CRS and extent.

---

## Key assumption

**The inference image and the COG must cover the same geographic extent.** The app
treats the inference image as a pixel-space rendering of the same scene as the COG:
it scales detection pixels by the width/height ratio between the two rasters and
maps them through the COG's geotransform. If the two files don't cover the same
area (or are cropped differently), the detected polygons will land in the wrong
place on the map.

---

## Project structure

```
.
├── server.py            # FastAPI backend - detection, georeferencing, tile server, API
├── requirements.txt
├── models/
│   └── yolo26m-seg.pt   # YOLO weights (auto-downloaded on first run if missing)
├── static/
│   └── index.html       # Single-page Leaflet frontend (map + side panel)
├── test/                # Sample inference image + matching COG to try the app
│   ├── boat_image.tif
│   └── boat_image_cog.tif
├── uploads/             # Uploaded files land here (auto-created)
└── output/              # GeoJSON results land here (auto-created)
```

### Main API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/detect` | Upload image + COG, start a detection job |
| `GET`  | `/api/job/{job_id}` | Poll job status / progress / results |
| `GET`  | `/api/classes` | Class options for the picker (`boat`, `surfboard`) |
| `GET`  | `/tiles/{job_id}/{z}/{x}/{y}.png` | XYZ tiles rendered from the COG |
| `GET`  | `/output/{job_id}_boats.geojson` | The detection GeoJSON (download) |

---

## Notes & limitations

This is a **school project**, scoped accordingly:

- **In-memory jobs** - job state lives in a Python dict and is **lost on restart**.
  There is no database or task queue (detection runs via FastAPI background tasks).
- **Single GPU / single process** - no batching across requests or multi-worker
  scaling; one detection at a time is the intended usage.
- **Same-extent assumption** - see above; there's no automatic check that the
  inference image and COG actually align.
- **No upload size limit / chunking** - very large uploads may be slow or time out.
- Detections are limited to the `boat` and `surfboard` classes by design.
