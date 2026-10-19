import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp


# ============================================================
# AIVA - GESTURE ENGINE
# Gesture & Interaction Module
# ============================================================

APP_NAME = "AIVA // GESTURE ENGINE"
WINDOW_NAME = "AIVA - Gesture Engine"

# Keep the model in the same folder as this Python file.
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# ----------------------- CONFIG -----------------------------

MAX_HANDS = 2
STABLE_FRAMES = 6
COOLDOWN_SECONDS = 0.90
CALIBRATION_SECONDS = 2.5
EVENT_HISTORY_SIZE = 5
WRIST_TRAIL_SIZE = 12

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30


# -------------------- HAND CONNECTIONS ----------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---------------------- GESTURES ----------------------------

GESTURE_COLORS = {
    "OPEN PALM": (255, 220, 80),
    "FIST": (80, 80, 255),
    "POINTING": (80, 255, 180),
    "PEACE": (255, 100, 220),
    "THUMBS UP": (80, 255, 100),
    "THUMBS DOWN": (80, 150, 255),
    "OK": (255, 180, 60),
    "OTHER GESTURE": (150, 150, 150),
}

GESTURE_ACTIONS = {
    "OPEN PALM": "WAKE",
    "FIST": "STOP",
    "POINTING": "SELECT",
    "PEACE": "NEXT",
    "THUMBS UP": "CONFIRM",
    "THUMBS DOWN": "CANCEL",
    "OK": "ACCEPT",
}

GESTURE_GUIDE = [
    ("OPEN PALM", "All five fingers extended"),
    ("FIST", "All fingers folded"),
    ("POINTING", "Index finger extended"),
    ("PEACE", "Index + middle extended"),
    ("THUMBS UP", "Thumb pointing upward"),
    ("THUMBS DOWN", "Thumb pointing downward"),
    ("OK", "Thumb + index touching"),
]


# ---------------------- DATA MODEL --------------------------

@dataclass
class HandState:
    current_gesture: str = "OTHER GESTURE"
    stable_gesture: str = "OTHER GESTURE"
    candidate_gesture: str = "OTHER GESTURE"
    candidate_count: int = 0
    last_event_time: float = 0.0
    handedness: str = "UNKNOWN"
    tracking_score: float = 0.0
    wrist_trail: deque = None
    last_event_gesture: str = ""


# ---------------------- HELPERS -----------------------------

def distance(point_a, point_b):
    return (
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
    ) ** 0.5


def recognize_gesture(points):
    """
    Converts MediaPipe hand landmarks into one of the supported
    AIVA gestures.
    """

    wrist = points[0]

    # Finger extension checks.
    index_up = distance(points[8], wrist) > distance(points[6], wrist) * 1.15
    middle_up = distance(points[12], wrist) > distance(points[10], wrist) * 1.15
    ring_up = distance(points[16], wrist) > distance(points[14], wrist) * 1.15
    pinky_up = distance(points[20], wrist) > distance(points[18], wrist) * 1.15

    fingers_up = [index_up, middle_up, ring_up, pinky_up]
    other_fingers_folded = not any(fingers_up)

    # Palm-size-normalized OK gesture.
    palm_scale = max(distance(points[0], points[9]), 0.001)
    thumb_index_distance = distance(points[4], points[8])

    other_three_open = sum(
        [middle_up, ring_up, pinky_up]
    ) >= 2

    if thumb_index_distance < palm_scale * 0.45 and other_three_open:
        return "OK"

    if all(fingers_up):
        return "OPEN PALM"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINTING"

    # Thumb direction.
    if other_fingers_folded and points[4].y < points[2].y - 0.04:
        return "THUMBS UP"

    if other_fingers_folded and points[4].y > points[2].y + 0.04:
        return "THUMBS DOWN"

    if other_fingers_folded:
        return "FIST"

    return "OTHER GESTURE"


def get_handedness(result, hand_index):
    """Safely reads MediaPipe's left/right hand classification."""
    try:
        hand_info = result.handedness[hand_index][0]
        name = hand_info.category_name.upper()
        score = float(hand_info.score)
        return name, score
    except (AttributeError, IndexError, TypeError, ValueError):
        return "UNKNOWN", 0.0


def draw_panel(frame, x, y, width, height, border_color,
               fill=(18, 14, 30), alpha=0.88):
    """Draws a transparent HUD panel."""
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + width, y + height),
        fill,
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        border_color,
        2,
    )


def put_text(frame, text, position, scale=0.6,
             color=(230, 230, 240), thickness=1):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_progress_bar(frame, x, y, width, height, progress,
                      color=(100, 255, 170)):
    progress = max(0.0, min(1.0, progress))

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (45, 45, 55),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + int(width * progress), y + height),
        color,
        -1,
    )


def draw_hand_effect(frame, pixel_points, color):
    """Adds a subtle professional glow around the tracked hand."""
    if not pixel_points:
        return

    xs = [p[0] for p in pixel_points]
    ys = [p[1] for p in pixel_points]

    center_x = int(sum(xs) / len(xs))
    center_y = int(sum(ys) / len(ys))

    radius = max(
        55,
        int(
            (
                (max(xs) - min(xs)) ** 2
                + (max(ys) - min(ys)) ** 2
            ) ** 0.5 * 0.18
        ),
    )

    overlay = frame.copy()

    cv2.circle(
        overlay,
        (center_x, center_y),
        radius,
        color,
        2,
    )

    cv2.addWeighted(
        overlay,
        0.30,
        frame,
        0.70,
        0,
        frame,
    )


# ------------------- GESTURE ENGINE -------------------------

def update_hand_state(state, gesture, now):
    """
    Handles gesture stability and cooldown.

    Returns:
        event_generated: bool
    """

    state.current_gesture = gesture

    if gesture == state.candidate_gesture:
        state.candidate_count += 1
    else:
        state.candidate_gesture = gesture
        state.candidate_count = 1

    stable = state.candidate_count >= STABLE_FRAMES

    if not stable:
        return False

    if gesture == state.stable_gesture:
        return False

    state.stable_gesture = gesture

    if gesture == "OTHER GESTURE":
        return False

    if now - state.last_event_time < COOLDOWN_SECONDS:
        return False

    state.last_event_time = now
    return True


def create_event(hand_number, state, timestamp):
    gesture = state.stable_gesture

    return {
        "time": timestamp,
        "hand": hand_number,
        "side": state.handedness,
        "gesture": gesture,
        "action": GESTURE_ACTIONS.get(gesture, "NONE"),
    }


# ---------------------- UI DRAWING --------------------------

def draw_header(frame, width):
    draw_panel(
        frame,
        18,
        15,
        width - 36,
        70,
        (255, 190, 50),
    )

    put_text(
        frame,
        APP_NAME,
        (40, 47),
        0.85,
        (255, 220, 100),
        2,
    )

    put_text(
        frame,
        "● SYSTEM ONLINE  |  INTERACTION READY",
        (width - 405, 47),
        0.58,
        (100, 255, 170),
        2,
    )


def draw_status_panel(frame, x, y, width, height,
                      states, fps):
    draw_panel(
        frame,
        x,
        y,
        width,
        height,
        (255, 100, 220),
    )

    put_text(
        frame,
        "GESTURE ENGINE",
        (x + 12, y + 25),
        0.44,
        (255, 220, 100),
        2,
    )

    hands_count = len(states)

    if hands_count:
        status = "TRACKING ACTIVE"
        status_color = (100, 255, 170)
    else:
        status = "WAITING FOR HAND"
        status_color = (120, 180, 255)

    put_text(
        frame,
        status,
        (x + 12, y + 46),
        0.34,
        status_color,
        1,
    )

    put_text(
        frame,
        f"HANDS       {hands_count}/{MAX_HANDS}",
        (x + 12, y + 66),
        0.33,
    )

    put_text(
        frame,
        f"FPS         {fps:.0f}",
        (x + 150, y + 66),
        0.33,
    )

    put_text(
        frame,
        "STABILITY",
        (x + 12, y + 86),
        0.33,
        (220, 220, 235),
        1,
    )

    y_offset = y + 108

    if not states:
        put_text(
            frame,
            "Show your hand to begin",
            (x + 12, y_offset),
            0.32,
            (160, 160, 175),
        )
        return

    for hand_number, state in states.items():
        progress = min(
            state.candidate_count / STABLE_FRAMES,
            1.0,
        )

        label = (
            f"{state.handedness}  "
            f"{state.candidate_gesture}"
        )

        put_text(
            frame,
            label[:34],
            (x + 12, y_offset),
            0.30,
            GESTURE_COLORS.get(
                state.candidate_gesture,
                (180, 180, 180),
            ),
            1,
        )

        draw_progress_bar(
            frame,
            x + 18,
            y_offset + 9,
            width - 36,
            6,
            progress,
            GESTURE_COLORS.get(
                state.candidate_gesture,
                (180, 180, 180),
            ),
        )

        y_offset += 34


def draw_gesture_guide(frame, x, y, width, height):
    """Compact gesture guide so the camera/hand remains visible."""

    draw_panel(
        frame,
        x,
        y,
        width,
        height,
        (80, 180, 255),
        alpha=0.82,
    )

    put_text(
        frame,
        "GESTURES",
        (x + 12, y + 23),
        0.45,
        (255, 220, 100),
        1,
    )

    # Compact rows: gesture name + short action.
    compact_actions = {
        "OPEN PALM": "WAKE",
        "FIST": "STOP",
        "POINTING": "SELECT",
        "PEACE": "NEXT",
        "THUMBS UP": "CONFIRM",
        "THUMBS DOWN": "CANCEL",
        "OK": "ACCEPT",
    }

    row_height = 29
    start_y = y + 48

    for index, (name, _) in enumerate(GESTURE_GUIDE):
        row_y = start_y + index * row_height

        if row_y + 20 > y + height:
            break

        color = GESTURE_COLORS[name]

        cv2.circle(
            frame,
            (x + 13, row_y - 4),
            4,
            color,
            -1,
            cv2.LINE_AA,
        )

        put_text(
            frame,
            name,
            (x + 25, row_y),
            0.36,
            color,
            1,
        )

        put_text(
            frame,
            compact_actions[name],
            (x + width - 72, row_y),
            0.32,
            (205, 205, 220),
            1,
        )


def draw_current_gesture(frame, x, y, width, gesture,
                         hand_side, tracking_score):
    color = GESTURE_COLORS.get(
        gesture,
        (180, 180, 180),
    )

    draw_panel(
        frame,
        x,
        y,
        width,
        92,
        color,
    )

    put_text(
        frame,
        "CURRENT GESTURE",
        (x + 14, y + 23),
        0.38,
        (220, 220, 235),
        1,
    )

    put_text(
        frame,
        gesture,
        (x + 14, y + 51),
        0.60,
        color,
        2,
    )

    action = GESTURE_ACTIONS.get(
        gesture,
        "WAITING",
    )

    put_text(
        frame,
        f"EVENT: {action}",
        (x + 14, y + 70),
        0.34,
        (225, 225, 235),
        1,
    )

    score_text = (
        f"{hand_side}  |  TRACK CONF: "
        f"{tracking_score * 100:.0f}%"
    )

    put_text(
        frame,
        score_text,
        (x + 14, y + 86),
        0.30,
        (165, 165, 180),
        1,
    )


def draw_event_history(frame, x, y, width, events):
    height = 34 + EVENT_HISTORY_SIZE * 19

    draw_panel(
        frame,
        x,
        y,
        width,
        height,
        (100, 255, 170),
    )

    put_text(
        frame,
        "RECENT GESTURE EVENTS",
        (x + 12, y + 23),
        0.38,
        (255, 220, 100),
        2,
    )

    if not events:
        put_text(
            frame,
            "No accepted gesture events yet",
            (x + 12, y + 45),
            0.31,
            (160, 160, 175),
        )
        return

    for index, event in enumerate(events):
        row_y = y + 46 + index * 19

        put_text(
            frame,
            (
                f"{event['side']:<6} "
                f"{event['gesture']:<15} "
                f"-> {event['action']}"
            ),
            (x + 12, row_y),
            0.29,
            GESTURE_COLORS.get(
                event["gesture"],
                (210, 210, 220),
            ),
            1,
        )


def draw_footer(frame, width, height):
    put_text(
        frame,
        "AIVA GESTURE ENGINE",
        (25, height - 25),
        0.43,
        (150, 150, 165),
        1,
    )

    put_text(
        frame,
        "H HUD  |  G GUIDE  |  T TRAIL  |  R RESET  |  Q EXIT",
        (width - 135, height - 25),
        0.43,
        (190, 190, 205),
        1,
    )


def draw_wrist_trail(frame, points, color):
    """Draws a restrained motion trail behind the wrist."""
    if not points or len(points) < 2:
        return

    overlay = frame.copy()
    total = len(points)
    for i in range(1, total):
        p1 = points[i - 1]
        p2 = points[i]
        thickness = max(1, int(1 + 3 * i / total))
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)


def draw_hand_progress(frame, center, progress, color):
    """Circular stability indicator around each tracked hand."""
    progress = max(0.0, min(1.0, progress))
    cx, cy = center
    cv2.ellipse(
        frame, (cx, cy), (30, 30), -90, 0, 360,
        (45, 45, 55), 3, cv2.LINE_AA
    )
    cv2.ellipse(
        frame, (cx, cy), (30, 30), -90, 0, int(360 * progress),
        color, 3, cv2.LINE_AA
    )


def draw_no_hand_hint(frame, width, height):
    """Minimal center hint when no hand is visible."""
    cx = width // 2
    cy = height // 2
    put_text(frame, "SHOW HAND", (cx - 70, cy - 8), 0.55, (170, 170, 185), 1)
    put_text(frame, "Gesture tracking ready", (cx - 85, cy + 18), 0.32, (120, 120, 135), 1)


# ----------------------- MAIN -------------------------------

def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Place 'hand_landmarker.task' in the same folder as this file.")
        return

    # MediaPipe model.
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )

    # Camera.
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        print("Check camera permissions or CAMERA_INDEX.")
        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    screen_width = int(
        cv2.getWindowImageRect(WINDOW_NAME)[2]
    )

    screen_height = int(
        cv2.getWindowImageRect(WINDOW_NAME)[3]
    )

    if screen_width <= 0:
        screen_width = 1366

    if screen_height <= 0:
        screen_height = 768

    # Per-hand state.
    hand_states = {}

    # Recent accepted events.
    event_history = deque(
        maxlen=EVENT_HISTORY_SIZE
    )

    # Startup/calibration timer.
    start_time = time.time()

    # FPS calculation.
    previous_frame_time = time.time()
    fps = 0.0

    # User-controllable HUD layers.
    hud_visible = True
    guide_visible = True
    trail_visible = True
    calibration_done = False

    # Popup.
    popup_text = "SYSTEM READY"
    popup_until = time.time() + CALIBRATION_SECONDS

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as tracker:

        while True:
            frame_start = time.time()

            success, camera_frame = camera.read()

            if not success:
                print("ERROR: Could not read frame from camera.")
                break

            # Mirror camera like a normal webcam.
            camera_frame = cv2.flip(
                camera_frame,
                1,
            )

            original_height, original_width = (
                camera_frame.shape[:2]
            )

            # MediaPipe requires RGB.
            rgb_frame = cv2.cvtColor(
                camera_frame,
                cv2.COLOR_BGR2RGB,
            )

            timestamp_ms = int(
                (time.time() - start_time) * 1000
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = tracker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # Resize camera to display.
            frame = cv2.resize(
                camera_frame,
                (screen_width, screen_height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Slight cinematic darkening.
            frame = cv2.convertScaleAbs(
                frame,
                alpha=0.70,
                beta=-8,
            )

            # Coordinate conversion.
            scale_x = screen_width / original_width
            scale_y = screen_height / original_height

            current_states = {}
            detected_hands = []

            now = time.time()
            calibration_progress = min((now - start_time) / CALIBRATION_SECONDS, 1.0)
            calibration_done = calibration_progress >= 1.0

            # ---------------- HAND PROCESSING ----------------

            if result.hand_landmarks:

                for hand_index, points in enumerate(
                    result.hand_landmarks
                ):

                    hand_number = hand_index + 1

                    gesture = recognize_gesture(
                        points
                    )

                    handedness, tracking_score = (
                        get_handedness(
                            result,
                            hand_index,
                        )
                    )

                    if hand_number not in hand_states:
                        hand_states[hand_number] = HandState(
                            wrist_trail=deque(maxlen=WRIST_TRAIL_SIZE)
                        )
                    elif hand_states[hand_number].wrist_trail is None:
                        hand_states[hand_number].wrist_trail = deque(maxlen=WRIST_TRAIL_SIZE)

                    state = hand_states[hand_number]

                    state.handedness = handedness
                    state.tracking_score = tracking_score

                    # Store a short wrist trail for a subtle motion visual.
                    wrist_px = (
                        int(points[0].x * original_width * scale_x),
                        int(points[0].y * original_height * scale_y),
                    )
                    state.wrist_trail.append(wrist_px)

                    event_generated = update_hand_state(
                        state,
                        gesture,
                        now,
                    )

                    # Create an accepted gesture event.
                    if event_generated and calibration_done:
                        event = create_event(
                            hand_number,
                            state,
                            time.strftime("%H:%M:%S"),
                        )

                        event_history.appendleft(
                            event
                        )
                        state.last_event_gesture = event["gesture"]

                        popup_text = (
                            f"{event['gesture']}  |  "
                            f"{event['action']}"
                        )

                        popup_until = (
                            now + 1.8
                        )

                    current_states[hand_number] = state

                    pixel_points = [
                        (
                            int(
                                point.x
                                * original_width
                                * scale_x
                            ),
                            int(
                                point.y
                                * original_height
                                * scale_y
                            ),
                        )
                        for point in points
                    ]

                    detected_hands.append(
                        (
                            hand_number,
                            state,
                            pixel_points,
                        )
                    )

            else:
                # No hands: keep history, but reset current
                # recognition state so a new gesture must stabilize.
                hand_states.clear()

            # ---------------- DRAW HANDS ---------------------

            for (
                hand_number,
                state,
                pixel_points,
            ) in detected_hands:

                color = GESTURE_COLORS.get(
                    state.current_gesture,
                    (180, 180, 180),
                )

                if trail_visible:
                    draw_wrist_trail(frame, state.wrist_trail, color)

                draw_hand_effect(
                    frame,
                    pixel_points,
                    color,
                )

                # Outer skeleton.
                for point_a, point_b in HAND_CONNECTIONS:

                    cv2.line(
                        frame,
                        pixel_points[point_a],
                        pixel_points[point_b],
                        (25, 25, 35),
                        9,
                        cv2.LINE_AA,
                    )

                    cv2.line(
                        frame,
                        pixel_points[point_a],
                        pixel_points[point_b],
                        color,
                        3,
                        cv2.LINE_AA,
                    )

                # Landmark points.
                for x, y in pixel_points:

                    cv2.circle(
                        frame,
                        (x, y),
                        8,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )

                # Stability ring around the wrist.
                wrist_x, wrist_y = pixel_points[0]
                stability_progress = min(
                    state.candidate_count / STABLE_FRAMES,
                    1.0,
                )
                draw_hand_progress(
                    frame,
                    (wrist_x, wrist_y),
                    stability_progress,
                    color,
                )

                # Hand label.
                label_y = max(
                    wrist_y - 25,
                    100,
                )

                put_text(
                    frame,
                    (
                        f"{state.handedness}  |  "
                        f"{state.current_gesture}"
                    ),
                    (
                        max(20, wrist_x - 70),
                        label_y,
                    ),
                    0.52,
                    color,
                    2,
                )

            # -------------------- UI -------------------------

            draw_header(
                frame,
                screen_width,
            )

            # ------------------------------------------------
            # COMPACT HUD LAYOUT
            # Keep the center of the screen free for the hand.
            # H = HUD | G = Guide | T = Trail | R = Reset
            # ------------------------------------------------

            guide_width = 285
            guide_x = screen_width - guide_width - 18

            if not current_states and calibration_done:
                draw_no_hand_hint(frame, screen_width, screen_height)

            if hud_visible:
                if guide_visible:
                    guide_height = min(260, screen_height - 430)
                    draw_gesture_guide(
                        frame,
                        guide_x,
                        98,
                        guide_width,
                        guide_height,
                    )

                status_y = 368
                status_height = 128

                draw_status_panel(
                    frame,
                    guide_x,
                    status_y,
                    guide_width,
                    status_height,
                    current_states,
                    fps,
                )

                left_width = min(350, max(270, guide_x - 45))

                if current_states:
                    first_hand_number = sorted(current_states.keys())[0]
                    first_state = current_states[first_hand_number]

                    draw_current_gesture(
                        frame,
                        20,
                        98,
                        left_width,
                        first_state.stable_gesture
                        if first_state.stable_gesture != "OTHER GESTURE"
                        else first_state.current_gesture,
                        first_state.handedness,
                        first_state.tracking_score,
                    )

                draw_event_history(
                    frame,
                    20,
                    205,
                    left_width,
                    list(event_history),
                )

            # Calibration indicator.
            if not calibration_done:
                cal_width = 360
                cal_x = (screen_width - cal_width) // 2
                cal_y = screen_height - 125
                draw_panel(frame, cal_x, cal_y, cal_width, 72, (255, 190, 50))
                put_text(frame, "CALIBRATING GESTURE ENGINE", (cal_x + 18, cal_y + 25), 0.40, (255, 220, 100), 1)
                draw_progress_bar(frame, cal_x + 18, cal_y + 42, cal_width - 36, 8, calibration_progress, (255, 190, 50))

            # Popup.
            if now < popup_until and calibration_done:

                popup_color = (
                    GESTURE_COLORS.get(
                        popup_text.split("  |  ")[0],
                        (255, 220, 100),
                    )
                )

                popup_width = min(
                    330,
                    max(250, guide_x - 55),
                )

                popup_x = 20
                popup_y = screen_height - 135

                draw_panel(
                    frame,
                    popup_x,
                    popup_y,
                    popup_width,
                    92,
                    popup_color,
                )

                put_text(
                    frame,
                    "GESTURE EVENT ACCEPTED",
                    (popup_x + 12, popup_y + 27),
                    0.38,
                    (220, 220, 235),
                    1,
                )

                put_text(
                    frame,
                    popup_text,
                    (popup_x + 12, popup_y + 63),
                    0.55,
                    popup_color,
                    2,
                )

            # FPS update.
            current_time = time.time()

            frame_delta = (
                current_time
                - previous_frame_time
            )

            previous_frame_time = current_time

            if frame_delta > 0:
                instant_fps = 1.0 / frame_delta
                fps = (
                    0.90 * fps
                    + 0.10 * instant_fps
                )

            draw_footer(
                frame,
                screen_width,
                screen_height,
            )

            # ---------------- DISPLAY ------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("h"):
                hud_visible = not hud_visible

            elif key == ord("g"):
                guide_visible = not guide_visible

            elif key == ord("t"):
                trail_visible = not trail_visible

            elif key == ord("r"):
                hand_states.clear()
                event_history.clear()
                popup_text = "SYSTEM RESET"
                popup_until = time.time() + 1.2
                start_time = time.time()
                calibration_done = False

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
