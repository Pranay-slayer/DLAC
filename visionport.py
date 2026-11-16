import base64
import io
import json
import os
import tempfile
import threading
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO


MODEL_NAME = os.getenv("VISIONPORT_MODEL", "yolov8s.pt")
IMAGE_SIZE = int(os.getenv("VISIONPORT_IMAGE_SIZE", "960"))
CONFIDENCE = float(os.getenv("VISIONPORT_CONFIDENCE", "0.2"))
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRAFFIC_CLASSES = {0, 1, 2, 3, 5, 7, 9}

app = FastAPI(title="VisionPort Traffic Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost", "http://127.0.0.1",
                   "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-VisionPort-Summary"],
)

print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
inference_lock = threading.Lock()
print("YOLO model loaded successfully.")


def decode_image(contents):
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return image


def as_detection_list(result):
    detections = []
    if result.boxes is None:
        return detections
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in TRAFFIC_CLASSES:
            continue
        x1, y1, x2, y2 = [round(float(value), 1) for value in box.xyxy[0].tolist()]
        track_id = None
        if box.id is not None:
            track_id = int(box.id[0].item())
        detections.append({
            "class_id": class_id,
            "label": names.get(class_id, str(class_id)),
            "confidence": round(float(box.conf[0].item()), 3),
            "box": [x1, y1, x2, y2],
            "track_id": track_id,
        })
    return detections


def image_signal_status(image, detections):
    lights = [item for item in detections if item["class_id"] == 9]
    if not lights:
        return "Not detected"

    color_names = []
    for light in lights:
        x1, y1, x2, y2 = [int(value) for value in light["box"]]
        crop = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if crop.size == 0:
            color_names.append("Unknown")
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red = int(np.count_nonzero(((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) &
                                   (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 70)))
        yellow = int(np.count_nonzero((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) &
                                      (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 100)))
        green = int(np.count_nonzero((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) &
                                     (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)))
        color_names.append(max(
            (("Red", red), ("Yellow", yellow), ("Green", green)),
            key=lambda item: item[1]
        )[0] if max(red, yellow, green) > 0 else "Unknown")
    return ", ".join(color_names)


def image_analysis(image, confidence):
    with inference_lock:
        results = model.predict(
            source=image, imgsz=IMAGE_SIZE, conf=confidence, device=None, verbose=False
        )
    result = results[0]
    detections = as_detection_list(result)
    counts = Counter(item["label"] for item in detections
                     if item["class_id"] in VEHICLE_CLASSES)
    total = sum(counts.values())
    density = "Low" if total <= 5 else "Medium" if total <= 12 else "High"
    annotated = result.plot()
    encoded_ok, encoded = cv2.imencode(".jpg", annotated)
    if not encoded_ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")
    return {
        "annotated_image": "data:image/jpeg;base64," +
                           base64.b64encode(encoded.tobytes()).decode("ascii"),
        "detections": detections,
        "vehicle_counts": dict(counts),
        "vehicle_count": total,
        "traffic_density": density,
        "traffic_signal": image_signal_status(image, detections),
        "limitations": [
            "Speed, waiting time, entry/exit, wrong-way/turn, and parking require a video and camera-specific calibration.",
            "Signal color and multiple-rider results are visual estimates, not safety-critical decisions."
        ],
    }, encoded


@app.get("/")
async def home():
    return {"status": "Active", "message": "VisionPort traffic analysis API is running."}


@app.get("/health")
async def health():
    return {
        "status": "Active",
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "device": "CUDA" if __import__("torch").cuda.is_available() else "CPU",
    }


@app.post("/analyze/")
async def analyze_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE, ge=0.05, le=0.9),
):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 200 MB limit.")
    image = decode_image(contents)
    analysis, _ = image_analysis(image, confidence)
    return analysis


@app.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE, ge=0.05, le=0.9),
):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 200 MB limit.")
    image = decode_image(contents)
    _, encoded = image_analysis(image, confidence)
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


def remove_file(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as error:
        print(f"Could not remove temporary video {path}: {error}")


def parse_parking_zone(value):
    if not value:
        return None
    try:
        points = [
            [float(coordinate) for coordinate in point.split(",")]
            for point in value.split(";")
        ]
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Parking zone must be normalized x,y points separated by semicolons.",
        ) from error
    if len(points) < 3 or any(
        len(point) != 2 or not all(0 <= coordinate <= 1 for coordinate in point)
        for point in points
    ):
        raise HTTPException(
            status_code=400,
            detail="Parking zone needs at least three normalized points between 0 and 1.",
        )
    return np.array(points, dtype=np.float32)


@app.post("/analyze-video/")
async def analyze_video(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE, ge=0.05, le=0.9),
    meters_per_pixel: float = Query(default=0.05, gt=0, le=2),
    expected_direction: str = Query(default="unspecified", pattern="^(unspecified|left|right|up|down)$"),
    parking_seconds: float = Query(default=10, ge=2, le=3600),
    allowed_turns: str = Query(default="all", pattern="^(all|straight-only|left-only|right-only)$"),
    parking_zone: str = Query(default=""),
):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 200 MB limit.")

    source_path = None
    output_path = None
    try:
        parking_polygon = parse_parking_zone(parking_zone)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as source:
            source.write(contents)
            source_path = source.name
        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Could not read this video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 120 else 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Video has invalid dimensions.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output:
            output_path = output.name
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Could not create annotated video.")

        tracks = defaultdict(list)
        class_peaks = Counter()
        peak_vehicle_count = 0
        frame_count = 0
        crossings = {"entries": 0, "exits": 0}
        last_side = {}
        signal_statuses = Counter()
        motorcycle_rider_max = defaultdict(int)

        try:
            with inference_lock:
                for result in model.track(
                    source=source_path, stream=True, persist=False,
                    imgsz=IMAGE_SIZE, conf=confidence, device=None, verbose=False,
                    tracker="bytetrack.yaml"
                ):
                    frame_count += 1
                    frame = result.orig_img
                    detections = as_detection_list(result)
                    vehicles = [item for item in detections
                                if item["class_id"] in VEHICLE_CLASSES]
                    peak_vehicle_count = max(peak_vehicle_count, len(vehicles))
                    frame_class_counts = Counter(item["label"] for item in vehicles)
                    for label, count in frame_class_counts.items():
                        class_peaks[label] = max(class_peaks[label], count)
                    if any(item["class_id"] == 9 for item in detections):
                        signal_statuses[image_signal_status(frame, detections)] += 1

                    people = [item for item in detections if item["class_id"] == 0]
                    for motorcycle in (item for item in vehicles if item["class_id"] == 3):
                        mx1, my1, mx2, my2 = motorcycle["box"]
                        riders = 0
                        for person in people:
                            px = (person["box"][0] + person["box"][2]) / 2
                            py = (person["box"][1] + person["box"][3]) / 2
                            if mx1 - 0.1 * (mx2 - mx1) <= px <= mx2 + 0.1 * (mx2 - mx1) and \
                                    my1 - 0.7 * (my2 - my1) <= py <= my2 + 0.1 * (my2 - my1):
                                riders += 1
                        if motorcycle["track_id"] is not None:
                            motorcycle_rider_max[motorcycle["track_id"]] = max(
                                motorcycle_rider_max[motorcycle["track_id"]], riders
                            )

                    for item in vehicles:
                        track_id = item["track_id"]
                        if track_id is None:
                            continue
                        x1, y1, x2, y2 = item["box"]
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        tracks[track_id].append({
                            "frame": frame_count, "x": center_x, "y": center_y,
                            "label": item["label"], "box": item["box"]
                        })
                        side = center_x >= width / 2
                        if track_id in last_side and side != last_side[track_id]:
                            crossings["entries" if side else "exits"] += 1
                        last_side[track_id] = side
                    writer.write(result.plot())
        finally:
            capture.release()
            writer.release()

        unique_tracks = len(tracks)
        speeds = []
        waiting_ids = []
        wrong_way_ids = []
        parked_ids = []
        turns = []
        waiting_times = {}
        prohibited_turns = []
        for track_id, points in tracks.items():
            if len(points) < 2:
                continue
            start, end = points[0], points[-1]
            seconds = max((end["frame"] - start["frame"]) / fps, 1 / fps)
            dx, dy = end["x"] - start["x"], end["y"] - start["y"]
            speed_kmh = ((dx * dx + dy * dy) ** 0.5 / seconds) * meters_per_pixel * 3.6
            speeds.append(speed_kmh)
            longest_stationary_frames = 0
            stationary_start = None
            for previous, current in zip(points, points[1:]):
                gap = current["frame"] - previous["frame"]
                distance = ((current["x"] - previous["x"]) ** 2 +
                            (current["y"] - previous["y"]) ** 2) ** 0.5
                if gap <= 2 and distance <= 2.5:
                    stationary_start = previous["frame"] if stationary_start is None else stationary_start
                    longest_stationary_frames = max(
                        longest_stationary_frames, current["frame"] - stationary_start
                    )
                else:
                    stationary_start = None
            wait_seconds = round(longest_stationary_frames / fps, 1)
            if wait_seconds >= 1:
                waiting_times[track_id] = wait_seconds
            if wait_seconds >= parking_seconds:
                waiting_ids.append(track_id)
                if parking_polygon is not None:
                    normalized_center = (
                        end["x"] / width,
                        end["y"] / height,
                    )
                    polygon_pixels = parking_polygon * np.array([width, height], dtype=np.float32)
                    if cv2.pointPolygonTest(
                        polygon_pixels, normalized_center, False
                    ) >= 0:
                        parked_ids.append(track_id)
            if expected_direction != "unspecified":
                axis_value = dx if expected_direction in ("left", "right") else dy
                actual = "right" if dx > 0 else "left" if dx < 0 else \
                         "down" if dy > 0 else "up" if dy < 0 else "unspecified"
                if expected_direction in ("up", "down"):
                    actual = "down" if dy > 0 else "up" if dy < 0 else "unspecified"
                if axis_value and actual != expected_direction and seconds > 1:
                    wrong_way_ids.append(track_id)
            if len(points) >= 5:
                segments = []
                for a, b in zip(points, points[1:]):
                    vx, vy = b["x"] - a["x"], b["y"] - a["y"]
                    if abs(vx) + abs(vy) > 3:
                        segments.append((vx, vy))
                if len(segments) >= 4:
                    first = np.mean(segments[:2], axis=0)
                    last = np.mean(segments[-2:], axis=0)
                    cross = first[0] * last[1] - first[1] * last[0]
                    dot = first[0] * last[0] + first[1] * last[1]
                    angle = abs(np.degrees(np.arctan2(cross, dot)))
                    if angle > 35:
                        turn_direction = "right" if cross > 0 else "left"
                        turn = {
                            "track_id": track_id,
                            "direction": turn_direction,
                            "angle_degrees": round(float(angle)),
                        }
                        turns.append(turn)
                        if allowed_turns == "straight-only" or \
                                allowed_turns == turn_direction + "-only":
                            prohibited_turns.append(turn)

        density = "Low" if peak_vehicle_count <= 5 else \
                  "Medium" if peak_vehicle_count <= 12 else "High"
        congestion = "High congestion" if density == "High" else \
                     "Moderate traffic" if density == "Medium" else \
                     "No congestion detected"
        summary = {
            "frame_count": frame_count,
            "fps": round(fps, 2),
            "vehicle_counts_per_frame_peak": peak_vehicle_count,
            "vehicle_classes_detected": dict(class_peaks),
            "unique_tracked_vehicles": unique_tracks,
            "traffic_density": density,
            "congestion_alert": congestion,
            "average_speed_kmh_estimate": round(float(np.mean(speeds)), 1) if speeds else None,
            "waiting_vehicle_ids": waiting_ids,
            "estimated_wait_seconds_by_track": waiting_times,
            "stopped_in_configured_restricted_zone_ids": parked_ids,
            "entry_exit_estimate": crossings,
            "traffic_signal_statuses": dict(signal_statuses),
            "wrong_way_candidates": wrong_way_ids,
            "turning_tracks_candidates": turns,
            "prohibited_turn_candidates": prohibited_turns,
            "suspected_triple_riding_ids": [
                track_id for track_id, count in motorcycle_rider_max.items() if count > 2
            ],
            "calibration": {
                "meters_per_pixel": meters_per_pixel,
                "expected_direction": expected_direction,
                "parking_seconds": parking_seconds,
                "allowed_turns": allowed_turns,
                "parking_zone_normalized": parking_zone,
                "entry_exit_line": "vertical midpoint of image",
            },
            "caveats": [
                "Speed and counts are approximate; use a fixed camera and calibrate meters_per_pixel.",
                "Wrong turns require mapped lanes/allowed turns; turning tracks are candidates only.",
                "Parking candidates are only reported inside the configured restricted polygon; this does not establish a legal violation.",
                "Rider association and signal color are estimates; use a locally trained model for reliable decisions."
            ],
        }
        response_headers = {
            "X-VisionPort-Summary": json.dumps(summary, separators=(",", ":")),
            "Cache-Control": "no-store",
        }
        remove_file(source_path)
        return FileResponse(
            output_path, media_type="video/mp4", filename="visionport-annotated.mp4",
            headers=response_headers, background=BackgroundTask(remove_file, output_path)
        )
    except HTTPException:
        if source_path:
            remove_file(source_path)
        if output_path:
            remove_file(output_path)
        raise
    except Exception as error:
        if source_path:
            remove_file(source_path)
        if output_path:
            remove_file(output_path)
        print(f"Video analysis error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
