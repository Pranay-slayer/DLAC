# VisionPort traffic monitoring

VisionPort provides a browser interface and a local FastAPI service for traffic image/video analysis. It detects COCO classes supported by the model (cars, motorcycles, buses, trucks, people, bicycles, and traffic lights), annotates detections, and uses ByteTrack for video IDs.

## Run locally

Use Python 3.10 or later. From this folder, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python visionport.py
```

Open `visionport.html` in a browser. The service listens at `http://127.0.0.1:8000`. The first run downloads the default `yolov8s.pt` weights. A GPU-capable PyTorch install can be used if CUDA is available. To use different or locally trained weights, set `VISIONPORT_MODEL` to their path before starting the API. `VISIONPORT_IMAGE_SIZE` and `VISIONPORT_CONFIDENCE` can also be set to tune inference.

## Available workflows

- Capture a still from the camera, upload an image, and view labeled boxes, confidence scores, counts, density, and a traffic-light color estimate.
- Upload a video for annotated playback and tracking IDs. Save an individual frame from the source-video player.
- Review approximate peak vehicle count/density, track-derived speed, midpoint entry/exit crossings, waiting/stopped vehicles, direction/turn candidates, and suspected multiple riders.
- Adjust video confidence, meters-per-pixel, expected direction, allowed-turn rule, stopped-vehicle duration, and optional restricted parking polygon in the page. Enter parking polygon vertices as normalized `x,y` pairs separated by semicolons (for example `0.1,0.2;0.8,0.2;0.8,0.9`).

The API also exposes `/health`, `/analyze/`, the backwards-compatible JPEG endpoint `/detect/`, and `/analyze-video/`.

## Important accuracy limits

The default model is a general COCO model, not an India-specific traffic model. Smaller/distant or occluded vehicles, unusual camera angles, night scenes, and dense overlaps can reduce recall. Use a fixed camera, higher-quality locally trained weights, and representative validation footage for stronger results.

Speed is an estimate based on a user-supplied meters-per-pixel scale. Entry/exit currently uses a virtual line at the image midpoint. Wrong-way results require the expected direction to be configured; turn changes are candidates, not rule violations. A stopped vehicle alone cannot establish illegal parking, and rider association/traffic-signal color are visual estimates. Reliable wrong-turn, illegal-parking, signal, and triple-riding decisions require camera-specific zones/lanes and task-specific labeled training data. Do not use these estimates for enforcement or safety-critical decisions.

For better real-world accuracy, collect and label representative footage, split it by camera/location into train/validation/test sets, train a model with the relevant local classes, then set `VISIONPORT_MODEL` to the resulting weights.
