import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


DEFAULT_OUTPUT_PATH = "output.mp4"
DEFAULT_EVENTS_CSV = "events.csv"
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_PROCESS_EVERY = 3
DEFAULT_CONF_THRESHOLD = 0.50
DEFAULT_MIN_BOX_AREA_RATIO = 0.0008
DEFAULT_CROP_PADDING_RATIO = 0.07
DEFAULT_STATE_HOLD_SECONDS = 1.0
DEFAULT_NEAR_ZONE_SCALE = 1.35
DEFAULT_DEVICE = "auto"
PERSON_CLASS_ID = 0


class TableState(str, Enum):
    EMPTY = "EMPTY"
    APPROACH = "APPROACH"
    OCCUPIED = "OCCUPIED"


@dataclass(frozen=True)
class PipelineConfig:
    video_path: str
    output_path: str
    events_csv_path: str
    roi_json_arg: Optional[str]
    model_path: str
    device_mode: str
    process_every: int
    conf_threshold: float
    min_box_area_ratio: float
    crop_padding_ratio: float
    state_hold_seconds: float
    near_zone_scale: float


@dataclass(frozen=True)
class FrameInfo:
    index: int
    timestamp_sec: float


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class ZoneObservation:
    has_inside: bool
    has_near: bool
    detections: Tuple[Detection, ...]


@dataclass(frozen=True)
class StateTransition:
    frame: int
    time_sec: float
    event: str
    raw_state: str
    stable_state_before: str
    stable_state_after: str
    has_inside: int
    has_near: int


class VideoOpenError(RuntimeError):
    pass


class InvalidRoiError(RuntimeError):
    pass


class OccupancyStateMachine:
    """
    Стабилизирует переходы между EMPTY / APPROACH / OCCUPIED.
    Новое состояние должно удерживаться минимум hold_seconds,
    чтобы считаться подтвержденным.
    """

    def __init__(self, hold_seconds: float) -> None:
        self.hold_seconds = hold_seconds
        self.state: Optional[TableState] = None
        self.pending_state: Optional[TableState] = None
        self.pending_since: Optional[float] = None

    def current_state_or_default(self) -> TableState:
        return self.state if self.state is not None else TableState.EMPTY

    def update(self, raw_state: TableState, t_sec: float) -> Optional[Tuple[TableState, TableState]]:
        """
        Возвращает кортеж (prev_state, new_state), если стабильное состояние изменилось.
        Иначе возвращает None.
        """
        if self.state is None:
            self.state = raw_state
            return TableState.EMPTY, raw_state

        if raw_state == self.state:
            self.pending_state = None
            self.pending_since = None
            return None

        if self.pending_state != raw_state:
            self.pending_state = raw_state
            self.pending_since = t_sec
            return None

        if self.pending_since is None:
            self.pending_since = t_sec
            return None

        if t_sec - self.pending_since < self.hold_seconds:
            return None

        prev_state = self.state
        self.state = raw_state
        self.pending_state = None
        self.pending_since = None
        return prev_state, raw_state


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Table occupancy prototype for one table on one video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output video")
    parser.add_argument("--events-csv", default=DEFAULT_EVENTS_CSV, help="Path to events CSV")
    parser.add_argument("--roi-json", default=None, help="ROI polygon as JSON string or path to JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to YOLO model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument("--process-every", type=int, default=DEFAULT_PROCESS_EVERY)
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--min-box-area-ratio", type=float, default=DEFAULT_MIN_BOX_AREA_RATIO)
    parser.add_argument("--crop-padding-ratio", type=float, default=DEFAULT_CROP_PADDING_RATIO)
    parser.add_argument("--state-hold-seconds", type=float, default=DEFAULT_STATE_HOLD_SECONDS)
    parser.add_argument("--near-zone-scale", type=float, default=DEFAULT_NEAR_ZONE_SCALE)
    args = parser.parse_args()

    if args.process_every < 1:
        raise ValueError("--process-every must be >= 1")
    if not 0.0 <= args.conf_threshold <= 1.0:
        raise ValueError("--conf-threshold must be in [0, 1]")
    if args.min_box_area_ratio < 0.0:
        raise ValueError("--min-box-area-ratio must be >= 0")
    if args.crop_padding_ratio < 0.0:
        raise ValueError("--crop-padding-ratio must be >= 0")
    if args.state_hold_seconds < 0.0:
        raise ValueError("--state-hold-seconds must be >= 0")
    if args.near_zone_scale < 1.0:
        raise ValueError("--near-zone-scale must be >= 1.0")

    return PipelineConfig(
        video_path=args.video,
        output_path=args.output,
        events_csv_path=args.events_csv,
        roi_json_arg=args.roi_json,
        model_path=args.model,
        device_mode=args.device,
        process_every=args.process_every,
        conf_threshold=args.conf_threshold,
        min_box_area_ratio=args.min_box_area_ratio,
        crop_padding_ratio=args.crop_padding_ratio,
        state_hold_seconds=args.state_hold_seconds,
        near_zone_scale=args.near_zone_scale,
    )


def select_device(mode: str) -> str:
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def show_progress(current: int, total: int) -> None:
    if total <= 0:
        return
    width = 28
    ratio = current / total
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    sys.stdout.write(f"\r[{bar}] {ratio * 100:6.2f}% ({current}/{total})")
    sys.stdout.flush()


def select_polygon_roi(frame: np.ndarray) -> np.ndarray:
    points: List[Tuple[int, int]] = []
    window = "Draw ROI polygon"

    def on_mouse(event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        preview = frame.copy()

        for px, py in points:
            cv2.circle(preview, (px, py), 4, (0, 255, 255), -1)

        if len(points) >= 2:
            cv2.polylines(preview, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
        if len(points) >= 3:
            cv2.polylines(preview, [np.array(points, dtype=np.int32)], True, (0, 255, 255), 2)

        cv2.putText(
            preview,
            "LMB:add  R:undo  C:clear  Enter:confirm",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window, preview)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("r"), ord("R")) and points:
            points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()
        elif key in (13, 10) and len(points) >= 3:
            break

    cv2.destroyWindow(window)
    return np.array(points, dtype=np.int32)


def load_polygon_from_json(roi_json_arg: Optional[str], width: int, height: int) -> Optional[np.ndarray]:
    if roi_json_arg is None:
        return None

    if os.path.exists(roi_json_arg):
        with open(roi_json_arg, "r", encoding="utf-8") as file:
            raw = file.read()
    else:
        raw = roi_json_arg

    data = json.loads(raw)
    points = data.get("points", []) if isinstance(data, dict) else data
    polygon = np.array(points, dtype=np.float32)

    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise InvalidRoiError("ROI polygon must contain at least 3 points of shape [x, y]")

    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return polygon.astype(np.int32)


def expand_polygon(polygon: np.ndarray, scale: float, frame_w: int, frame_h: int) -> np.ndarray:
    center = np.mean(polygon.astype(np.float32), axis=0)
    expanded = center + (polygon.astype(np.float32) - center) * scale
    expanded[:, 0] = np.clip(expanded[:, 0], 0, frame_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, frame_h - 1)
    return expanded.astype(np.int32)


def bottom_center_in_polygon(bbox_xyxy: Tuple[int, int, int, int], polygon: np.ndarray) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    bx = (x1 + x2) / 2.0
    by = float(y2)
    return cv2.pointPolygonTest(polygon, (bx, by), False) >= 0


def compute_raw_state(has_inside: bool, has_near: bool) -> TableState:
    if has_inside:
        return TableState.OCCUPIED
    if has_near:
        return TableState.APPROACH
    return TableState.EMPTY


def build_transition(
    frame_info: FrameInfo,
    raw_state: TableState,
    prev_state: TableState,
    new_state: TableState,
    observation: ZoneObservation,
) -> StateTransition:
    return StateTransition(
        frame=frame_info.index,
        time_sec=frame_info.timestamp_sec,
        event=new_state.value,
        raw_state=raw_state.value,
        stable_state_before=prev_state.value,
        stable_state_after=new_state.value,
        has_inside=int(observation.has_inside),
        has_near=int(observation.has_near),
    )


def transitions_to_dataframe(transitions: Sequence[StateTransition]) -> pd.DataFrame:
    if not transitions:
        return pd.DataFrame(
            columns=[
                "frame",
                "time_sec",
                "event",
                "raw_state",
                "stable_state_before",
                "stable_state_after",
                "has_inside",
                "has_near",
            ]
        )

    df = pd.DataFrame([t.__dict__ for t in transitions])
    return df.sort_values(["time_sec", "frame"]).reset_index(drop=True)


def compute_empty_to_next_engagement_delays(events_df: pd.DataFrame) -> List[float]:
    """
    Для каждого EMPTY ищем первое следующее событие из:
    - APPROACH
    - OCCUPIED

    Это лучше соответствует формулировке:
    "через какое время к столу подошел следующий человек".
    """
    if events_df.empty:
        return []

    delays: List[float] = []
    waiting_empty_time: Optional[float] = None

    for _, row in events_df.sort_values(["time_sec", "frame"]).iterrows():
        event = str(row["event"])
        time_sec = float(row["time_sec"])

        if event == TableState.EMPTY.value:
            waiting_empty_time = time_sec
            continue

        if waiting_empty_time is None:
            continue

        if event in (TableState.APPROACH.value, TableState.OCCUPIED.value) and time_sec > waiting_empty_time:
            delays.append(time_sec - waiting_empty_time)
            waiting_empty_time = None

    return delays


def make_crop_rect(
    polygon: np.ndarray,
    frame_w: int,
    frame_h: int,
    crop_padding_ratio: float,
) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(polygon)
    if w <= 0 or h <= 0:
        raise InvalidRoiError("Invalid ROI polygon bounding box")

    pad = int(min(frame_w, frame_h) * crop_padding_ratio)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w, x + w + pad)
    y2 = min(frame_h, y + h + pad)

    if x2 <= x1 or y2 <= y1:
        raise InvalidRoiError("Invalid crop area around ROI")

    return x1, y1, x2, y2


def run_detector(
    model: YOLO,
    frame: np.ndarray,
    crop_rect: Tuple[int, int, int, int],
    conf_threshold: float,
    min_box_area: float,
) -> Tuple[Detection, ...]:
    cx1, cy1, cx2, cy2 = crop_rect
    crop = frame[cy1:cy2, cx1:cx2]

    result = model(crop, verbose=False, classes=[PERSON_CLASS_ID], conf=conf_threshold)[0]
    detections: List[Detection] = []

    if result.boxes is None or len(result.boxes) == 0:
        return tuple(detections)

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes), dtype=np.float32)

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box.astype(np.int32)
        det = Detection(
            x1=int(x1 + cx1),
            y1=int(y1 + cy1),
            x2=int(x2 + cx1),
            y2=int(y2 + cy1),
            confidence=float(conf),
        )
        if det.area < min_box_area:
            continue
        detections.append(det)

    return tuple(detections)


def analyze_zones(
    detections: Sequence[Detection],
    table_polygon: np.ndarray,
    near_polygon: np.ndarray,
) -> ZoneObservation:
    has_inside = False
    has_near = False

    for det in detections:
        inside = bottom_center_in_polygon(det.xyxy, table_polygon)
        near = bottom_center_in_polygon(det.xyxy, near_polygon)
        has_inside = has_inside or inside
        has_near = has_near or near

    return ZoneObservation(
        has_inside=has_inside,
        has_near=has_near,
        detections=tuple(detections),
    )


def draw_overlay(
    frame: np.ndarray,
    table_polygon: np.ndarray,
    near_polygon: np.ndarray,
    state: TableState,
    detections: Sequence[Detection],
    avg_delay: Optional[float],
) -> np.ndarray:
    rendered = frame.copy()
    x, y, _, _ = cv2.boundingRect(table_polygon)

    if state == TableState.OCCUPIED:
        roi_color = (0, 0, 255)
    elif state == TableState.APPROACH:
        roi_color = (0, 165, 255)
    else:
        roi_color = (0, 255, 0)

    for det in detections:
        cv2.rectangle(rendered, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 255), 2)
        label = f"person {det.confidence:.2f}"
        cv2.putText(
            rendered,
            label,
            (det.x1, max(20, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    cv2.polylines(rendered, [table_polygon], True, roi_color, 2)
    cv2.polylines(rendered, [near_polygon], True, (170, 170, 170), 1)

    cv2.putText(
        rendered,
        f"state={state.value}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        roi_color,
        2,
    )

    avg_text = f"avg_delay={avg_delay:.2f}s" if avg_delay is not None else "avg_delay=N/A"
    cv2.putText(
        rendered,
        avg_text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return rendered


def print_final_report(events_df: pd.DataFrame, delays: Sequence[float]) -> None:
    print(f"Events: {len(events_df)}")
    if delays:
        print(f"EMPTY->NEXT_ENGAGEMENT pairs: {len(delays)}")
        print(f"Average delay: {float(np.mean(delays)):.2f} sec")
        print(f"Median delay: {float(np.median(delays)):.2f} sec")
        print(f"Min delay: {float(np.min(delays)):.2f} sec")
        print(f"Max delay: {float(np.max(delays)):.2f} sec")
    else:
        print("No EMPTY->(APPROACH|OCCUPIED) pairs found")


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    config = parse_args()

    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None

    try:
        cap = cv2.VideoCapture(config.video_path)
        if not cap.isOpened():
            raise VideoOpenError(f"Cannot open video: {config.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise VideoOpenError("Invalid FPS in input video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = max(1, width * height)

        writer = cv2.VideoWriter(
            config.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {config.output_path}")

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise VideoOpenError("Cannot read first frame")

        table_polygon = load_polygon_from_json(config.roi_json_arg, width, height)
        if table_polygon is None:
            table_polygon = select_polygon_roi(first_frame)

        near_polygon = expand_polygon(table_polygon, config.near_zone_scale, width, height)
        crop_rect = make_crop_rect(table_polygon, width, height, config.crop_padding_ratio)

        device = select_device(config.device_mode)
        model = YOLO(config.model_path)
        model.to(device)

        logger.info("Video: %s", config.video_path)
        logger.info("Output: %s", config.output_path)
        logger.info("Events CSV: %s", config.events_csv_path)
        logger.info("Device: %s", device)

        min_box_area = frame_area * config.min_box_area_ratio
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fsm = OccupancyStateMachine(config.state_hold_seconds)
        transitions: List[StateTransition] = []

        cached_detections: Tuple[Detection, ...] = tuple()
        cached_observation = ZoneObservation(False, False, tuple())

        frame_idx = 0
        last_progress_ts = 0.0
        avg_delay_live: Optional[float] = None

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % config.process_every == 0:
                cached_detections = run_detector(
                    model=model,
                    frame=frame,
                    crop_rect=crop_rect,
                    conf_threshold=config.conf_threshold,
                    min_box_area=min_box_area,
                )
                cached_observation = analyze_zones(
                    detections=cached_detections,
                    table_polygon=table_polygon,
                    near_polygon=near_polygon,
                )

            frame_info = FrameInfo(
                index=frame_idx,
                timestamp_sec=frame_idx / fps,
            )

            raw_state = compute_raw_state(
                has_inside=cached_observation.has_inside,
                has_near=cached_observation.has_near,
            )

            transition_result = fsm.update(raw_state, frame_info.timestamp_sec)
            current_state = fsm.current_state_or_default()

            if transition_result is not None:
                prev_state, new_state = transition_result
                transition = build_transition(
                    frame_info=frame_info,
                    raw_state=raw_state,
                    prev_state=prev_state,
                    new_state=new_state,
                    observation=cached_observation,
                )
                transitions.append(transition)
                logger.info(
                    "[EVENT] t=%.2fs frame=%d %s -> %s",
                    transition.time_sec,
                    transition.frame,
                    transition.stable_state_before,
                    transition.stable_state_after,
                )

                tmp_df = transitions_to_dataframe(transitions)
                tmp_delays = compute_empty_to_next_engagement_delays(tmp_df)
                avg_delay_live = float(np.mean(tmp_delays)) if tmp_delays else None

            rendered = draw_overlay(
                frame=frame,
                table_polygon=table_polygon,
                near_polygon=near_polygon,
                state=current_state,
                detections=cached_detections,
                avg_delay=avg_delay_live,
            )
            writer.write(rendered)
            frame_idx += 1

            now = time.monotonic()
            if now - last_progress_ts >= 5:
                show_progress(frame_idx, total_frames)
                last_progress_ts = now

        if total_frames > 0:
            show_progress(frame_idx, total_frames)
            print()

        events_df = transitions_to_dataframe(transitions)
        events_df.to_csv(config.events_csv_path, index=False)

        delays = compute_empty_to_next_engagement_delays(events_df)
        logger.info("Saved events to %s", config.events_csv_path)
        print_final_report(events_df, delays)

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()