import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Дефолтный путь к выходному видео.
DEFAULT_OUTPUT_PATH = "output.mp4"
# Дефолтный путь к CSV с событиями.
DEFAULT_EVENTS_CSV = "events.csv"
# Дефолтная YOLO-модель.
DEFAULT_MODEL_PATH = "yolov8n.pt"
# Запускать инференс раз в N кадров.
DEFAULT_PROCESS_EVERY = 3
# Минимальный confidence детекции.
DEFAULT_CONF_THRESHOLD = 0.5
# Минимальная площадь bbox как доля от площади кадра.
DEFAULT_MIN_BOX_AREA_RATIO = 0.0008
# Отступ вокруг ROI для crop-инференса.
DEFAULT_CROP_PADDING_RATIO = 0.07
# Антидребезг в секундах.
DEFAULT_STATE_HOLD_SECONDS = 1.0
# Масштаб near-зоны вокруг столика.
DEFAULT_NEAR_ZONE_SCALE = 1.20
# Максимальное расширение near-зоны от границы стола в пикселях.
DEFAULT_NEAR_ZONE_MAX_OFFSET_PX = 70.0
# Устройство по умолчанию.
DEFAULT_DEVICE = "auto"
# Rule for table/near presence estimation.
DEFAULT_PRESENCE_RULE = "hybrid"
# Event timestamps: "frame" (legacy) or "inference" (accurate when process_every > 1).
DEFAULT_EVENT_TIME_SOURCE = "inference"
# IoA thresholds for overlap between person bbox and table/near masks.
DEFAULT_TABLE_IOA_THRESHOLD = 0.10
DEFAULT_NEAR_IOA_THRESHOLD = 0.03
# Только person.
PERSON_CLASS_ID = 0


class OccupancyStateMachine:
    """FSM для стабилизации состояний стола и отсечения кратковременного шума детекции."""
    # Стабилизирует переходы EMPTY/APPROACH/OCCUPIED.
    def __init__(self, hold_seconds: float):
        """Инициализирует таймер удержания, чтобы смена состояния подтверждалась только после задержки."""
        # Сколько секунд нужно удерживать новое состояние,
        # чтобы оно стало "стабильным".
        self.hold_seconds = hold_seconds
        # Текущее подтвержденное состояние.
        self.state = None
        # Кандидат на следующее состояние (еще не подтвержден).
        self.pending_state = None
        # Время, когда pending_state впервые появился.
        self.pending_since = None

    def update(self, raw_state: str, t_sec: float):
        """Принимает сырое состояние и время кадра; возвращает True только при подтверждённой смене состояния."""
        # Возвращает True, если стабильное состояние реально изменилось на этом кадре.
        if self.state is None:
            self.state = raw_state
            return True
        if raw_state == self.state:
            self.pending_state = None
            self.pending_since = None
            return False
        if self.pending_state != raw_state:
            self.pending_state = raw_state
            self.pending_since = t_sec
            return False
        if t_sec - self.pending_since < self.hold_seconds:
            return False
        self.state = raw_state
        self.pending_state = None
        self.pending_since = None
        return True


def parse_args():
    """Читает параметры запуска, чтобы гибко настраивать обработку без правки исходников."""
    # CLI нужен для повторяемых прогонов (особенно в headless/CI),
    # чтобы не править код руками под каждый запуск.
    parser = argparse.ArgumentParser(description="Table cleanup event prototype")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output video")
    parser.add_argument("--events-csv", default=DEFAULT_EVENTS_CSV, help="Path to output CSV")
    parser.add_argument("--roi-json", default=None, help="ROI polygon as JSON string or path to JSON file")
    parser.add_argument("--near-roi-json", default=None, help="Near-zone polygon as JSON string or path to JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument("--process-every", type=int, default=DEFAULT_PROCESS_EVERY)
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--min-box-area-ratio", type=float, default=DEFAULT_MIN_BOX_AREA_RATIO)
    parser.add_argument("--crop-padding-ratio", type=float, default=DEFAULT_CROP_PADDING_RATIO)
    parser.add_argument("--state-hold-seconds", type=float, default=DEFAULT_STATE_HOLD_SECONDS)
    parser.add_argument("--near-zone-scale", type=float, default=DEFAULT_NEAR_ZONE_SCALE)
    parser.add_argument("--near-zone-max-offset-px", type=float, default=DEFAULT_NEAR_ZONE_MAX_OFFSET_PX)
    return parser.parse_args()


def select_device(mode: str):
    """Выбирает CPU/GPU для инференса в одном месте, чтобы не дублировать эту логику по коду."""
    # Выбор устройства в одном месте, чтобы не дублировать логику в коде инференса.
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def show_progress(cur: int, total: int):
    """Печатает прогресс-бар, чтобы видеть ход долгой обработки видео в консоли."""
    # Легкий текстовый прогресс-бар без внешних зависимостей.
    if total <= 0:
        return
    width = 28
    p = cur / total
    done = int(width * p)
    bar = "#" * done + "-" * (width - done)
    sys.stdout.write(f"\r[{bar}] {p*100:6.2f}% ({cur}/{total})")
    sys.stdout.flush()


def select_polygon_roi(
    frame,
    window_title="Draw ROI polygon",
    prompt_text="LMB: add  R: undo  C: clear  Enter: confirm",
    reference_polygon=None,
):
    """Интерактивно рисует полигон ROI мышкой; нужен для ручной разметки зоны стола/near без JSON."""
    # Интерактивный выбор ROI полигона.
    # Используется как fallback, если --roi-json не передан.
    points = []
    window = window_title

    def on_mouse(event, x, y, flags, param):
        """Обработчик клика: добавляет вершину полигона."""
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        preview = frame.copy()
        if reference_polygon is not None and len(reference_polygon) >= 3:
            cv2.polylines(preview, [reference_polygon.astype(np.int32)], True, (255, 0, 0), 2)
            cv2.putText(preview, "blue = table ROI", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        if points:
            for px, py in points:
                cv2.circle(preview, (px, py), 4, (0, 255, 255), -1)
        if len(points) >= 2:
            cv2.polylines(preview, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
        if len(points) >= 3:
            cv2.polylines(preview, [np.array(points, dtype=np.int32)], True, (0, 255, 255), 2)
        cv2.putText(preview, prompt_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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


def load_polygon_from_json(roi_json_arg, width, height):
    """Загружает полигон из JSON-строки/файла и ограничивает точки границами кадра."""
    # Поддержка двух форматов:
    # 1) JSON-строка: [[x1,y1], [x2,y2], ...]
    # 2) путь к JSON-файлу с тем же содержимым или {"points": [...]}
    if roi_json_arg is None:
        return None
    if os.path.exists(roi_json_arg):
        with open(roi_json_arg, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = roi_json_arg
    data = json.loads(raw)
    if isinstance(data, dict):
        points = data.get("points", [])
    else:
        points = data
    polygon = np.array(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("ROI polygon must contain at least 3 [x, y] points")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return polygon.astype(np.int32)


def expand_polygon(polygon, scale, frame_w, frame_h, max_offset_px=None):
    """Расширяет полигон стола в near-зону, чтобы фиксировать подход до входа в сам стол."""
    # Строим near-zone геометрическим масштабированием от центра полигона.
    # Нужна для состояния APPROACH (человек рядом, но еще не в ROI стола).
    polygon_f = polygon.astype(np.float32)
    center = np.mean(polygon_f, axis=0)
    deltas = (polygon_f - center) * (scale - 1.0)

    if max_offset_px is not None and max_offset_px > 0:
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-6)
        factors = np.minimum(1.0, float(max_offset_px) / safe_norms)
        deltas = deltas * factors

    expanded = polygon_f + deltas
    expanded[:, 0] = np.clip(expanded[:, 0], 0, frame_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, frame_h - 1)
    return expanded.astype(np.int32)


def bottom_center_in_polygon(bbox_xyxy, polygon):
    """Проверяет попадание нижнего центра bbox в полигон как устойчивую опорную точку."""
    # Более устойчивая проверка для человека:
    # используем нижнюю центральную точку bbox (условная "точка опоры").
    # Это уменьшает ложные срабатывания, когда широкий bbox задевает стол краем.
    x1, y1, x2, y2 = bbox_xyxy
    bx = (x1 + x2) / 2.0
    by = float(y2)
    return cv2.pointPolygonTest(polygon, (bx, by), False) >= 0


def anchor_center_in_polygon(bbox_xyxy, polygon, y_ratio):
    """Проверяет произвольную опорную точку bbox по высоте для более точной логики inside/near."""
    x1, y1, x2, y2 = bbox_xyxy
    bx = (x1 + x2) / 2.0
    by = float(y1) + float(y2 - y1) * float(y_ratio)
    return cv2.pointPolygonTest(polygon, (bx, by), False) >= 0


def build_polygon_mask(frame_h, frame_w, polygon):
    """Строит бинарную маску полигона; нужна для расчёта IoA-пересечения."""
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def box_ioa_with_mask(bbox_xyxy, mask):
    """Считает долю площади bbox, попавшую в маску; метрика используется в правилах presence."""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = mask.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box_area = float((x2 - x1) * (y2 - y1))
    if box_area <= 0:
        return 0.0
    overlap_area = float(cv2.countNonZero(mask[y1:y2, x1:x2]))
    return overlap_area / box_area


def classify_box_presence(
    bbox_xyxy,
    table_polygon,
    near_polygon,
    table_mask,
    near_mask,
    presence_rule,
    table_ioa_threshold,
    near_ioa_threshold,
):
    """Определяет для одного человека признаки inside/near по выбранному правилу."""
    table_bottom = bottom_center_in_polygon(bbox_xyxy, table_polygon)
    near_bottom = bottom_center_in_polygon(bbox_xyxy, near_polygon)
    table_anchor = anchor_center_in_polygon(bbox_xyxy, table_polygon, y_ratio=0.75)
    near_anchor = anchor_center_in_polygon(bbox_xyxy, near_polygon, y_ratio=0.90)
    if presence_rule == "bottom_center":
        inside = table_bottom
        near = near_bottom and not inside
        return inside, near
    table_ioa = box_ioa_with_mask(bbox_xyxy, table_mask)
    near_ioa = box_ioa_with_mask(bbox_xyxy, near_mask)
    if presence_rule == "ioa":
        inside = table_ioa >= table_ioa_threshold
        near = near_ioa >= near_ioa_threshold and not inside
        return inside, near
    # "hybrid": OCCUPIED подтверждаем только когда есть и IoA, и опорная точка в столе.
    # Это снижает ложное OCCUPIED у людей, стоящих рядом с краем стола.
    inside = table_anchor and table_ioa >= table_ioa_threshold
    near = (near_anchor or near_ioa >= near_ioa_threshold or near_bottom) and not inside
    return inside, near


def compute_raw_state(has_inside, has_near, stable_state):
    """Преобразует признаки inside/near в сырое состояние стола с поддержкой APPROACH после EMPTY."""
    # Приоритет у OCCUPIED:
    # если человек в ROI стола -> OCCUPIED,
    # иначе если в near-zone -> APPROACH,
    # иначе EMPTY.
    if not has_inside and not has_near:
        return "EMPTY"
    if stable_state in (None, "EMPTY"):
        # APPROACH = first appearance after EMPTY.
        return "APPROACH"
    if has_inside:
        return "OCCUPIED"
    return "APPROACH"


def compute_departure_to_next_person_delays(events_df: pd.DataFrame):
    """Считает задержки от ухода гостя (OCCUPIED->EMPTY) до следующего появления человека."""
    # Отсчет начинаем только когда стол стал EMPTY после OCCUPIED.
    # Останавливаем отсчет на первом следующем APPROACH или OCCUPIED.
    delays = []
    waiting_departure_time = None
    for _, row in events_df.sort_values(["time_sec", "frame"]).iterrows():
        event = row["event"]
        stable_before = row.get("stable_state_before", "")
        t_sec = float(row["time_sec"])
        if event == "EMPTY" and stable_before == "OCCUPIED":
            waiting_departure_time = t_sec
        elif event in ("APPROACH", "OCCUPIED") and waiting_departure_time is not None and t_sec > waiting_departure_time:
            delays.append(t_sec - waiting_departure_time)
            waiting_departure_time = None
    return delays


def main():
    """Главный пайплайн: видео, детекция, FSM состояний, отрисовка и выгрузка CSV событий."""
    # Основной пайплайн:
    # 1) открыть видео и подготовить writer,
    # 2) получить ROI (JSON или интерактивно),
    # 3) прогнать детекцию людей,
    # 4) обновить FSM и лог событий,
    # 5) сохранить видео и аналитику.
    args = parse_args()
    if args.process_every < 1:
        raise ValueError("--process-every must be >= 1")

    presence_rule = DEFAULT_PRESENCE_RULE
    event_time_source = DEFAULT_EVENT_TIME_SOURCE
    table_ioa_threshold = DEFAULT_TABLE_IOA_THRESHOLD
    near_ioa_threshold = DEFAULT_NEAR_IOA_THRESHOLD

    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError("Invalid FPS in input video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = max(1, width * height)

        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {args.output}")

        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError("Cannot read first frame")

        # ROI можно передать JSON-ом (строка/файл) или выбрать интерактивно.
        table_polygon = load_polygon_from_json(args.roi_json, width, height)
        if table_polygon is None:
            table_polygon = select_polygon_roi(first, window_title="Draw TABLE polygon")

        near_polygon = load_polygon_from_json(args.near_roi_json, width, height)
        if near_polygon is None:
            near_polygon = expand_polygon(
                table_polygon,
                args.near_zone_scale,
                width,
                height,
                args.near_zone_max_offset_px,
            )
        table_mask = None
        near_mask = None
        if presence_rule in ("ioa", "hybrid"):
            table_mask = build_polygon_mask(height, width, table_polygon)
            near_outer_mask = build_polygon_mask(height, width, near_polygon)
            # near_mask как "кольцо" вокруг стола, чтобы APPROACH не смешивался с OCCUPIED.
            near_mask = cv2.subtract(near_outer_mask, table_mask)
        x, y, rw, rh = cv2.boundingRect(table_polygon)
        if rw <= 0 or rh <= 0:
            raise RuntimeError("Invalid ROI polygon")
        nx, ny, nrw, nrh = cv2.boundingRect(near_polygon)
        if nrw <= 0 or nrh <= 0:
            raise RuntimeError("Invalid near polygon")

        pad = int(min(width, height) * args.crop_padding_ratio)
        # Для скорости считаем модель только на crop вокруг near-зоны.
        # Это снижает нагрузку и ускоряет обработку.
        cx1 = max(0, nx - pad)
        cy1 = max(0, ny - pad)
        cx2 = min(width, nx + nrw + pad)
        cy2 = min(height, ny + nrh + pad)

        device = select_device(args.device)
        model = YOLO(args.model)
        model.to(device)
        print(f"Using device: {device}")

        min_box_area = frame_area * args.min_box_area_ratio
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fsm = OccupancyStateMachine(args.state_hold_seconds)
        event_rows = []
        delays = []
        frame_idx = 0
        last_prog = 0.0
        cached_boxes = []
        cached_inside = False
        cached_near = False
        last_infer_frame_idx = 0
        last_infer_t_sec = 0.0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % args.process_every == 0:
                # Инференс делаем не на каждом кадре, а раз в N кадров.
                # На промежуточных кадрах используем последние кэшированные результаты.
                crop = frame[cy1:cy2, cx1:cx2]
                res = model(crop, verbose=False, classes=[PERSON_CLASS_ID], conf=args.conf_threshold)[0]

                cached_boxes = []
                cached_inside = False
                cached_near = False
                last_infer_frame_idx = frame_idx
                last_infer_t_sec = frame_idx / fps

                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy().astype(np.int32)
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        x1 += cx1
                        x2 += cx1
                        y1 += cy1
                        y2 += cy1
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        if area < min_box_area:
                            # Мелкие bbox часто являются шумом/ложными срабатываниями.
                            continue
                        cached_boxes.append((x1, y1, x2, y2))
                        inside, near = classify_box_presence(
                            (x1, y1, x2, y2),
                            table_polygon,
                            near_polygon,
                            table_mask,
                            near_mask,
                            presence_rule,
                            table_ioa_threshold,
                            near_ioa_threshold,
                        )
                        cached_inside = cached_inside or inside
                        cached_near = cached_near or near

            if event_time_source == "inference":
                t_sec = last_infer_t_sec
                event_frame_idx = last_infer_frame_idx
            else:
                t_sec = frame_idx / fps
                event_frame_idx = frame_idx
            stable_before = fsm.state if fsm.state is not None else "EMPTY"
            raw_state = compute_raw_state(cached_inside, cached_near, fsm.state)
            changed = fsm.update(raw_state, t_sec)
            stable_after = fsm.state if fsm.state is not None else "EMPTY"

            if changed:
                # Пишем диагностику перехода в CSV.
                # Эти поля помогают разбирать ложные или спорные события.
                event_rows.append(
                    {
                        "frame": event_frame_idx,
                        "time_sec": t_sec,
                        "event": stable_after,
                        "raw_state": raw_state,
                        "stable_state_before": stable_before,
                        "stable_state_after": stable_after,
                        "has_inside": int(cached_inside),
                        "has_near": int(cached_near),
                        "event_time_source": event_time_source,
                    }
                )
                print(f"[EVENT] t={t_sec:.2f}s -> {stable_after}")

                # Один источник аналитики: постобработка event_rows той же функцией.
                events_df_tmp = pd.DataFrame(event_rows)
                delays = compute_departure_to_next_person_delays(events_df_tmp) if not events_df_tmp.empty else []

            for x1, y1, x2, y2 in cached_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "person", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            draw_state = fsm.state if fsm.state is not None else "EMPTY"
            if draw_state == "OCCUPIED":
                roi_color = (0, 0, 255)
            elif draw_state == "APPROACH":
                roi_color = (0, 165, 255)
            else:
                roi_color = (0, 255, 0)
            cv2.polylines(frame, [table_polygon], True, roi_color, 2)
            cv2.polylines(frame, [near_polygon], True, (170, 170, 170), 1)
            cv2.putText(frame, f"state={draw_state}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

            avg_delay = float(np.mean(delays)) if delays else None
            avg_text = f"avg_delay={avg_delay:.2f}s" if avg_delay is not None else "avg_delay=N/A"
            cv2.putText(frame, avg_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(frame)
            frame_idx += 1

            now = time.monotonic()
            if now - last_prog >= 5:
                show_progress(frame_idx, total_frames)
                last_prog = now

        if total_frames > 0:
            show_progress(frame_idx, total_frames)
            print()

        events_df = pd.DataFrame(event_rows)
        if not events_df.empty:
            events_df = events_df.sort_values(["time_sec", "frame"]).reset_index(drop=True)
        events_df.to_csv(args.events_csv, index=False)
        print(f"Events: {len(events_df)} | saved: {args.events_csv}")

        final_delays = compute_departure_to_next_person_delays(events_df) if not events_df.empty else []
        if final_delays:
            print(f"Departure->next-person pairs: {len(final_delays)}")
            print(f"Average delay: {float(np.mean(final_delays)):.2f} sec")
        else:
            print("No departure->next-person pairs found")

    finally:
        # Гарантированный cleanup даже при исключениях в середине обработки.
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
