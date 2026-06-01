#!/usr/bin/env python3
"""View and capture video from the Android IP Webcam app.

Typical IP Webcam app address:
    http://192.168.1.23:8080

Live video endpoint used by this script:
    http://192.168.1.23:8080/video
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_WEAPON_LABELS = {
    "knife",
    "scissors",
    "baseball bat",
    "gun",
    "pistol",
    "rifle",
    "shotgun",
    "firearm",
    "weapon",
}


@dataclass
class DetectionConfig:
    enabled: bool
    confidence: float
    model_path: str
    weapon_labels: set[str]
    alert_dir: Path
    alert_cooldown: float


@dataclass
class WeaponDetection:
    label: str
    confidence: float


def normalize_base_url(address: str) -> str:
    """Return a clean base URL from an IP, host:port, or full URL."""
    address = address.strip()
    if not address:
        raise ValueError("address cannot be empty")

    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"

    parsed = urlparse(address)
    if not parsed.netloc:
        raise ValueError(f"invalid address: {address}")

    return address.rstrip("/")


def check_connection(base_url: str, timeout: float = 4.0) -> None:
    """Check that the phone responds before opening the video stream."""
    snapshot_url = f"{base_url}/shot.jpg"
    request = urllib.request.Request(snapshot_url, headers={"User-Agent": "Python IP Webcam Viewer"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ConnectionError(f"camera returned HTTP {response.status}")
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"could not reach {snapshot_url}. Make sure the phone and computer "
            "are on the same Wi-Fi network and the IP Webcam server is running."
        ) from exc


def save_frame(cv2, frame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"ip-webcam-{timestamp}.jpg"

    if not cv2.imwrite(str(path), frame):
        raise OSError(f"failed to save snapshot to {path}")

    return path


def parse_labels(labels: str) -> set[str]:
    return {label.strip().lower() for label in labels.split(",") if label.strip()}


def load_detector(config: DetectionConfig):
    if not config.enabled:
        return None

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        print(
            "Missing dependency: ultralytics. Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return None

    print(f"Loading detection model: {config.model_path}")
    detector = YOLO(config.model_path)
    model_labels = {str(name).lower() for name in detector.names.values()}
    missing_labels = sorted(config.weapon_labels - model_labels)

    if missing_labels:
        print(
            "Warning: this model does not contain these requested labels: "
            f"{', '.join(missing_labels)}"
        )
        print("For real gun detection, use a custom YOLO weapon model with gun/pistol classes.")

    return detector


def draw_weapon_detections(cv2, frame, detector, config: DetectionConfig) -> list[WeaponDetection]:
    if detector is None:
        return []

    detections = []
    results = detector.predict(frame, conf=config.confidence, verbose=False)

    for result in results:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = str(names[class_id]).lower()

            if label not in config.weapon_labels:
                continue

            confidence = float(box.conf[0])
            detections.append(WeaponDetection(label=label, confidence=confidence))
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Possible weapon: {label} {confidence:.2f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    if detections:
        cv2.putText(
            frame,
            "ALERT: POSSIBLE WEAPON",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return detections


def handle_weapon_alert(cv2, frame, detections: list[WeaponDetection], config: DetectionConfig) -> Path:
    labels = ", ".join(f"{item.label} {item.confidence:.2f}" for item in detections)
    print(f"\aALERT: possible weapon detected: {labels}")
    return save_frame(cv2, frame, config.alert_dir)


def view_stream(
    base_url: str,
    output_dir: Path,
    skip_check: bool,
    detection_config: DetectionConfig,
) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print(
            "Missing dependency: opencv-python. Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    detector = load_detector(detection_config)

    if not skip_check:
        check_connection(base_url)

    video_url = f"{base_url}/video"
    camera = cv2.VideoCapture(video_url)

    if not camera.isOpened():
        print(f"Could not open video stream: {video_url}", file=sys.stderr)
        return 1

    print("Connected.")
    print("Controls: press 's' to save a snapshot, 'q' or Esc to quit.")
    if detection_config.enabled and detector is not None:
        print(
            "Weapon detection is enabled for these labels: "
            f"{', '.join(sorted(detection_config.weapon_labels))}"
        )

    last_alert_time = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Lost connection or no frame received.", file=sys.stderr)
                return 1

            detections = draw_weapon_detections(cv2, frame, detector, detection_config)
            now = time.monotonic()
            if detections and now - last_alert_time >= detection_config.alert_cooldown:
                path = handle_weapon_alert(cv2, frame, detections, detection_config)
                print(f"Saved alert frame: {path}")
                last_alert_time = now

            cv2.imshow("IP Webcam Viewer", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                return 0

            if key == ord("s"):
                path = save_frame(cv2, frame, output_dir)
                print(f"Saved {path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to an Android IP Webcam app stream over Wi-Fi."
    )
    parser.add_argument(
        "address",
        help="Phone camera address, for example 192.168.1.23:8080 or http://192.168.1.23:8080",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("snapshots"),
        help="Folder where snapshots are saved when you press 's'.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Open the video stream without first checking /shot.jpg.",
    )
    parser.add_argument(
        "--detect-weapons",
        action="store_true",
        help="Draw alerts for possible weapons detected in the video stream.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum detection confidence from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model file to use. The default may download on first run.",
    )
    parser.add_argument(
        "--weapon-labels",
        default=",".join(sorted(DEFAULT_WEAPON_LABELS)),
        help="Comma-separated model labels that should trigger weapon alerts.",
    )
    parser.add_argument(
        "--alert-dir",
        type=Path,
        default=Path("alerts"),
        help="Folder where alert frames are saved.",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=3.0,
        help="Seconds to wait between printed/saved alerts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        base_url = normalize_base_url(args.address)
        detection_config = DetectionConfig(
            enabled=args.detect_weapons,
            confidence=args.confidence,
            model_path=args.model,
            weapon_labels=parse_labels(args.weapon_labels),
            alert_dir=args.alert_dir,
            alert_cooldown=args.alert_cooldown,
        )
        return view_stream(base_url, args.output_dir, args.skip_check, detection_config)
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
