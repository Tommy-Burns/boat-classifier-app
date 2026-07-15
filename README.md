This is the official repository for the **BOAT CLASSIFIER APP** developed by **Thomas Burns Botchwey** and **Ee Faye Chong**


|          |                                        |
| ----------------- | ------------------------------------------- |
| **Course**        | Application Development (Earth Observation) |
| **Course Number** | 651.052                                     |
| **Semester**      | Summer Semester 2026                        |
| **School**        | University of Salzburg, Austria             |

# Repository Structure
```
.
├── 📂 annotator            # discarded, this was not used in the app
├── 📂 colab_notebooks      # Google Colab notebooks converted to python files
├── 📂 yolo-boat-classifier # the boat classifier app
├── 🖹 README.md             # this file
```

- The annotator tool was intended to be used for creating segments with extra fields so that we use it to fine tune the YOLO models to perform our semantic segmentation. But the YOLO model we fine-tuned with our segments did not work, it detected almost no boats. We believe it is due to the fact that we could not get many labels from the image to fine-tune. And so we discarded the annotator app.
- The colab notebooks folder contains the Google Colab notebooks where we experimented the methods before translating the best approach into the final app. They have been converted to regular Python files because huggingface tokens were being committed even after deleting them from the code cells.
- yolo-boat-classifier is the final app and has been referred in the rest of this document as "the app".

# About the app

Detect boats in high-resolution aerial / satellite imagery and view them as a
georeferenced overlay on an interactive map.

The app runs **SAHI (Slicing Aided Hyper Inference)** over a YOLO segmentation model,
converts the detected boat masks into polygons, georeferences them, and serves the result as a GeoJSON overlay
on a Leaflet map. A COG of the same image is streamed to the map as XYZ tiles by a small
built-in tile server.

---

## How it works

1. You upload two files: an **inference image** (what the model looks at) and a
   matching **georeferenced COG** (the map basemap).
2. The backend runs YOLO model on the inference image using Slicing Aided Hyper Inference and returns the masks.
3. Each boat mask is traced to a polygon, its pixel coordinates are mapped through
   the **inference image**'s geotransform, and the polygons are reprojected to WGS84 (EPSG:4326)
   so Leaflet can draw them in the right place.
4. The frontend renders the COG as map tiles, overlays the boat polygons, and
   gives you tools to filter, inspect, and export them.

### Why sliced inference (SAHI)?

Aerial / satellite scenes are huge - often tens of thousands of pixels.
Feeding a whole scene to the model at once either **runs out of GPU memory** or
forces such aggressive downscaling that **small boats disappear**. SAHI
(Slicing Aided Hyper Inference) cuts the image into manageable, overlapping
tiles (default 640×640), runs detection on each at full resolution, and stitches
the predictions back together. This keeps memory usage optimised
and preserves the detail needed to catch small vessels near the resolution
limit.

---

## Features

- **Configurable detection** - slice size, overlap ratio, and target classes
  (`boat` / `surfboard`) are available on the UI for the user to choose from.
- **Real per-slice progress** - the status reads e.g. `Processing slice 47 / 180`
  while inference runs, backed by a real progress bar.
- **Post-detection confidence slider** - detection runs once at a low confidence and
  returns every detection with its score. Then the user can filter for different confidence levels accordingly.
- **Colour-by-confidence** - polygons are shaded on a red - yellow - green gradient,
  with a legend on the map.
- **Boat list** - every detection is listed in the side panel. Click a row to zoom
  to that boat and open its popup. The list is sortable by the confidence.
- **Layer controls** - overlay opacity slider and a show/hide toggle so the imagery
  underneath stays visible.
- **GeoJSON export** - download the full GeoJSON of detections.

---

## Tech stack

| Layer | Tools |
| --- | --- |
| Backend / API | **FastAPI** |
| Detection | **Ultralytics YOLO26m-seg** wrapped by **SAHI** sliced inference |
| Georeferencing & tiles | **rasterio** (image reading / transforms), **pyproj** (CRS reprojection), **mercantile** (XYZ tiling) |
| Frontend | **Leaflet JS**, **HTML**, **CSS**, **JavaScript** |

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

---

# AI Use Declaration
_This declaration is made by Thomas Burns Botchwey_
| SN | Use | Reason |
|---|---|------|
| 1. | The test image was created with Google Gemini Nano Banana | We were not sure if the images can be uploaded to Google Colab for the processing. Also, the original image is a very high resolution image and was exhausting our Google Colab GPU compute so we needed a smaller file which can still be used to build the pipeline. So in other not to risk any licensing problem for online image sources (some of it were also not free), we resorted to Gemini to generate one which is similar and free to distribute. |
| 2. | Background jobs in the API | The sliding window effect introduces a virtual window on the inference image and performs the detection one after the other which means that for large images which takes a long time to run, the user needs some kind of progress updates without breaking the detection loop. Claude AI from Anthropic was used to help wrap the process around background jobs as that was beyond our present knowledge. |
| 3. | Frontend Styling | The frontend CSS styling was also done with the help of Claude AI from Anthropic |
