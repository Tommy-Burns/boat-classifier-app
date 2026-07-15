# Google Colab Files (Converted to regular Python files)

Four independent scripts, originally Colab notebooks. Each one is described on its own below.

## `sahi.py`

The script loads `yolo26m-seg.pt` through `AutoDetectionModel`, then calls `get_sliced_prediction` with 640×640 slices and 20% overlap.  Predictions are then filtered down to the classes `boat` and `surfboard` - surfboard is kept because COCO-trained models often label small vessels that way. The filtered list is written back onto the result and `export_visuals` saves an annotated image with a timestamped filename.

The second half turns the masks into GeoJSON. `rasterio.open` checks whether the image carries a CRS and affine transform; a GeoTIFF does, a PNG does not. For each mask, `cv2.findContours` pulls out the outline, contours with fewer than 3 points are skipped, and each contour becomes a Polygon feature carrying its class and confidence. The output is saved in `output/boats.geojson`. The image is now plotted with the geojsons on it as a visual check.

## `sam3.py`

This uses the SAM3 model and without any fine-tuning, runs inference on the image. `SAM3SemanticPredictor` allows to query [`boat`, `ship`, `vessel`, `watercraft`, `a boat on water`] from the predictions.

## `yolo_test.py`

This file was purposed to fine-tune the YOLO model using labels exported from the Annotator app (discarded - see the [**`README.md`**](https://github.com/Tommy-Burns/boat-classifier-app/blob/main/README.md) file in the repository root).  
It was not used because the output was detecting very little to no boats. We believe it is due to the fact that we could not get many labels from the image to fine-tune. And so we discarded the annotator app, and all of its associating logic in this script.  
It has a:
- `geojson_to_yolo` which converts the geojson fields into YOLO labels.
- `finetune_yolo` which then trains from the output of the `geojson_to_yolo` function.
- `sliding_window_detect` function which was intended to perform a sliding window detection just like <a target="_blank" href="https://obss.github.io/sahi/"><b>SAHI (Slicing Aided Hyper Inference)</b></a> does.

## `random_forest.py`

This one works on polygons exported from eCognition Software by Tremble Inc. Features are every remaining column except the labels and `geometry`; the target is `Classified_as_ship`. 

Using an an 80/20 split, it fits a `RandomForestClassifier` with 100 trees, prints accuracy, and plots `feature_importances_` as a bar chart against the column names.

