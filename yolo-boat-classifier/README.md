# Boat Detection App
The detailed description of this app is described in the README file in the main repository.

---

## Setup & installation

Recommended **Python 3.10** or higher.

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


---

## Model Sizes

The app lets you choose between five YOLO segmentation model sizes via the **Model Size** dropdown. They share the same architecture but differ in the number of parameters, which trades inference speed against detection accuracy. Larger models detect more boats (especially small or partial ones) but run slower and use more GPU memory.

| Size | Code | Relative speed | Relative accuracy | Best for |
|------|------|----------------|-------------------|----------|
| Nano | `n` | Fastest | Lowest | Quick tests, low-resource machines, large batches |
| Small | `s` | Fast | Lower | A step up from nano with modest cost |
| Medium | `m` | Balanced | Good | **Default** - the recommended balance of speed and quality |
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
# Activate the virtual environment if not already activated
python server.py
```

This starts Uvicorn on `http://0.0.0.0:8000`. Then open:

```
http://localhost:8000
```

---

## Try it out

A ready-to-use sample pair is included in the **`test/`** folder so you can see the
app working immediately:

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

---

## Key assumption

**The inference image and the COG must cover the same geographic extent.**  If the two files don't cover the same
area (or are cropped differently), the detected polygons will land in the wrong
place on the map.

---

## Project structure

```
.
├── 🖹 server.py            # FastAPI backend - detection, georeferencing, tile server, API
├── 🖹 requirements.txt
├── 📂 models/
│   └── 🖹 yolo26m-seg.pt   # YOLO weights (auto-downloaded on first run if missing)
├── 📂 static/
│   └── 🖹 index.html       # Single-page Leaflet frontend (map + side panel)
├── 📂 test/                # Sample inference image + matching COG to try the app
│   ├── 🖼️ boat_image.tif
│   └── 🖼️ boat_image_cog.tif
├── 📂 uploads/             # Uploaded files land here (auto-created at runtime)
└── 📂 output/              # GeoJSON results land here (auto-created at runtime)
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
- **No upload size limit / chunking** - very large uploads may be slow or time out.
- Detections are limited to the `boat` and `surfboard` classes by design.
