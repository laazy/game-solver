#!/usr/bin/env python3
"""Recognize and solve the colored arrow game from an Android screenshot.

Requires numpy and OpenCV. Run: python3 arrows/solve.py --device HOST:PORT --play
Without --play, prints the planned moves without touching the device.
"""
import argparse
import os
import math
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import shutil
import threading
import json
import traceback
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

HUES = [(0, 8), (8, 22), (22, 37), (37, 80), (80, 100), (100, 130), (130, 155), (155, 180)]
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TEMPLATE = cv2.imread(str(Path(__file__).with_name('arrow_template.png')), 0) // 255
TRANSITION_HISTORY = deque(maxlen=16)


class TexturedTheme(ValueError):
    """The board's color distribution indicates an animated texture."""


@dataclass
class Arrow:
    mask: np.ndarray
    x: int
    y: int
    direction: int
    score: float
    radius: int


def match_head(local, sizes):
    """Exhaustively find the best arrowhead template in one component."""
    best = (-1, None, None, None)
    for size in sizes:
        template = cv2.resize(TEMPLATE, (size, size), interpolation=cv2.INTER_NEAREST)
        if min(local.shape) < size:
            continue
        for direction in range(4):
            rotated = np.rot90(template, direction).copy()
            scores = cv2.matchTemplate(local, rotated, cv2.TM_CCOEFF_NORMED)
            _, value, _, point = cv2.minMaxLoc(scores)
            if value > best[0]:
                best = value, direction, point, size
    return best


def recognize(image, workers=None):
    height, width = image.shape[:2]
    image = cv2.resize(image, (1440, round(height * 1440 / width)))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    arrows = []
    board_size = None
    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for low, high in HUES:
            mask = ((hsv[:, :, 0] >= low) & (hsv[:, :, 0] < high) &
                    (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 100)).astype(np.uint8)
            # The game board occupies the central region; exclude interface controls.
            mask[:int(image.shape[0] * .15)] = 0
            mask[int(image.shape[0] * .89):] = 0
            mask[:, :30] = 0
            mask[:, 1410:] = 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            components = []
            for label in range(1, count):
                x, y, w, h, area = stats[label]
                if area < 40:
                    # Small isolated pixels are usually antialiasing noise,
                    # but a larger fragment might be an unmodeled arrow. Do
                    # not silently plan through it as though it were empty.
                    if area >= 5:
                        raise ValueError(f'Unrecognized small colored component at {x},{y}, area {area}')
                    continue
                x0, y0 = max(0, x - 25), max(0, y - 25)
                local = (labels[y0:y+h+25, x0:x+w+25] == label).astype(np.uint8)
                components.append((label, x, y, x0, y0, local))
            if not components:
                continue
            first_result = None
            if board_size is None:
                # The first component determines scale using the same exhaustive
                # search as before. Its result is retained rather than repeated.
                first_result = match_head(components[0][5], range(19, 68, 2))
                board_size = first_result[3]
                if board_size is None:
                    _, x, y, _, _, _ = components[0]
                    raise ValueError(f'Unrecognized component at {x},{y}: no matching template')
            sizes = range(max(19, board_size - 2), min(67, board_size + 2) + 1, 2)
            futures = [None if i == 0 and first_result is not None
                       else executor.submit(match_head, component[5], sizes)
                       for i, component in enumerate(components)]
            for i, (component, future) in enumerate(zip(components, futures)):
                label, x, y, x0, y0, _ = component
                score, direction, point, size = first_result if future is None else future.result()
                if score < .85:
                    raise ValueError(f'Unrecognized component at {x},{y}, score {score:.2f}, size {size}, board {board_size}')
                arrows.append(Arrow(labels == label, x0 + point[0] + size // 2,
                                    y0 + point[1] + size // 2, direction, score, size // 2))
    return arrows


def blockers_for(arrows):
    if not arrows:
        return []
    occupied = np.zeros(arrows[0].mask.shape, dtype=np.int32)
    for i, arrow in enumerate(arrows):
        occupied[arrow.mask] = i + 1
    blockers = []
    for i, arrow in enumerate(arrows):
        x, y, radius = arrow.x, arrow.y, arrow.radius
        if arrow.direction == 0:
            ray = occupied[max(0, y-5):y+6, :max(0, x-radius)]
        elif arrow.direction == 1:
            ray = occupied[y+radius:, max(0, x-5):x+6]
        elif arrow.direction == 2:
            ray = occupied[max(0, y-5):y+6, x+radius:]
        else:
            ray = occupied[:max(0, y-radius), max(0, x-5):x+6]
        blockers.append({int(value) - 1 for value in np.unique(ray) if value and value != i + 1})
    return blockers


def crosses_tap(moving, target):
    """Whether the moving head's forward path passes the other tap point."""
    dx, dy = target.x - moving.x, target.y - moving.y
    margin = max(moving.radius, target.radius)
    if moving.direction == 0:
        return dx < 0 and abs(dy) < margin
    if moving.direction == 1:
        return dy > 0 and abs(dx) < margin
    if moving.direction == 2:
        return dx > 0 and abs(dy) < margin
    return dy < 0 and abs(dx) < margin


def plan_waves(arrows, batch_size, blockers=None):
    if blockers is None:
        blockers = blockers_for(arrows)
    remaining = set(range(len(arrows)))
    waves = []
    while remaining:
        ready = [i for i in sorted(remaining) if not (blockers[i] & remaining)]
        if not ready:
            raise ValueError(f'No collision-free move among {len(remaining)} arrows; recognition or game rules may differ')
        wave = []
        for i in ready:
            if all(not crosses_tap(arrows[j], arrows[i]) for j in wave):
                wave.append(i)
            if len(wave) == batch_size:
                break
        waves.append(wave)
        remaining.difference_update(wave)
    return waves


def plan(arrows):
    return [i for wave in plan_waves(arrows, 1) for i in wave]


def screenshot(adb):
    data = subprocess.check_output(adb + ['exec-out', 'screencap', '-p'])
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Invalid ADB screenshot')
    return image


def flat_palette_fraction(board, colored_mask):
    """Fraction covered by the eight most frequent quantized colors."""
    pixels = (board[colored_mask] // 8).astype(np.uint32)
    if not len(pixels):
        return 0.0
    packed = (pixels[:, 0] << 10) | (pixels[:, 1] << 5) | pixels[:, 2]
    counts = np.bincount(packed, minlength=32768)
    return float(np.partition(counts, -8)[-8:].sum() / len(pixels))


def stable_board(adb, timeout=15, allow_ui=False):
    deadline = time.monotonic() + timeout
    previous = None
    textured_samples = 0
    while time.monotonic() < deadline:
        frame = screenshot(adb)
        if match_ui(frame, 'reward'):
            if not allow_ui:
                raise RunFailure('Reward dialog must be dismissed before read-only recognition', frame)
            tap(adb, frame, (720, 2600))
            time.sleep(.5)
            # Theme unlocks can switch to a textured theme. Restore the
            # color theme expected by the recognizer before solving.
            frame = screenshot(adb)
            tap(adb, frame, (370, 240))
            time.sleep(.5)
            frame = screenshot(adb)
            color = match_ui(frame, 'color_select', region=(550, 1100, 900, 1400))
            if color:
                tap(adb, frame, color)
            tap(adb, frame, (1290, 740))
            time.sleep(.5)
            previous = None
            continue
        small = cv2.resize(frame, (360, 800))
        board = small[160:710]
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        colored_mask = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 100)
        colored = np.count_nonzero(colored_mask)
        if colored > 500 and np.min(small[170, 10]) >= 240:
            # Flat color uses a small palette; textured or animated themes
            # contain hundreds of shades and must be switched before solving.
            flat_fraction = flat_palette_fraction(board, colored_mask)
            textured_samples = textured_samples + 1 if flat_fraction < .70 else 0
            if textured_samples >= 2:
                raise TexturedTheme('Textured theme detected')
        if colored > 500 and previous is not None:
            changed = np.mean(np.max(cv2.absdiff(board, previous), axis=2) > 15)
            if changed < .001:
                return frame
        previous = board
        time.sleep(.1)
    raise RuntimeError('Board did not become stable')


# UI elements are constrained to their known locations. Matching small regions
# at quarter resolution avoids scanning the entire screen for each control.
UI_REGIONS = {
    'reward': (350, 500, 1100, 820),
    'collection': (420, 2500, 1050, 2800),
    'knit': (460, 2680, 1000, 2940),
    'failure': (350, 900, 1100, 1200),
    'confirm': (400, 2100, 1040, 2450),
    'close': (1160, 550, 1420, 820),
    'next': (720, 1800, 1320, 2750),
    'victory': (450, 800, 990, 1150),
}


@lru_cache(maxsize=None)
def ui_template(name):
    template = cv2.imread(str(Path(__file__).with_name(name + '.png')))
    if template is None:
        raise ValueError(f'Missing UI template: {name}')
    return cv2.resize(template, (template.shape[1] // 4, template.shape[0] // 4))


def match_ui(image, name, region=None):
    image = cv2.resize(image, (360, round(image.shape[0] * 360 / image.shape[1])))
    offset_x = offset_y = 0
    region = region or UI_REGIONS.get(name)
    if region is not None:
        left, top, right, bottom = (value // 4 for value in region)
        image = image[top:bottom, left:right]
        offset_x, offset_y = left, top
    template = ui_template(name)
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return None
    _, score, _, pos = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED))
    if score < .75:
        return None
    point = (4 * (pos[0] + offset_x) + 2 * template.shape[1], 4 * (pos[1] + offset_y) + 2 * template.shape[0])
    if name == 'next':
        # Correlation can confuse similarly shaped orange share buttons with
        # the green next button. Require its expected position and color.
        candidate = image[pos[1]:pos[1] + template.shape[0], pos[0]:pos[0] + template.shape[1]]
        blue, green, red, _ = cv2.mean(candidate)
        if point[0] < 900 or green < red + 20 or green < blue + 20:
            return None
    return point


def board_visible(frame):
    """Require both a white canvas and readable hearts before declaring a board."""
    small = cv2.resize(frame, (360, round(frame.shape[0] * 360 / frame.shape[1])))
    if np.min(small[170, 10]) < 240 or count_hearts(frame) is None:
        return False
    hsv = cv2.cvtColor(small[160:710], cv2.COLOR_BGR2HSV)
    return np.count_nonzero((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 100)) > 500


def tap(adb, image, point):
    scale = image.shape[1] / 1440
    # Android's /system/bin/input is a shell wrapper around `cmd input`.
    # Invoke the same service directly for UI taps, avoiding that wrapper.
    subprocess.run(adb + ['shell', 'cmd', 'input', 'tap', str(round(point[0]*scale)), str(round(point[1]*scale))], check=True)


def knit_duration(frame, expected_amount=None):
    """Estimate the press time from the yarn counter; fall back safely."""
    if expected_amount:
        return min(5000, max(250, round((expected_amount / 44 + .15) * 1000)))
    if not shutil.which('tesseract'):
        return 2000
    image = cv2.resize(frame, (1440, round(frame.shape[0] * 1440 / frame.shape[1])))
    hsv = cv2.cvtColor(image[2580:2685, 600:840], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (180, 85, 255))
    ok, encoded = cv2.imencode('.png', 255 - mask)
    if not ok:
        return 2000
    try:
        result = subprocess.run(
            ['tesseract', 'stdin', 'stdout', '-l', 'eng', '--psm', '7',
             '-c', 'tessedit_char_whitelist=0123456789'],
            input=encoded.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=2, check=True)
        digits = result.stdout.decode().strip()
        if digits and int(digits) < 1000:
            amount = int(digits)
            return 0 if amount == 0 else min(5000, max(250, round((amount / 44 + .15) * 1000)))
    except (OSError, subprocess.SubprocessError):
        pass
    return 2000


def advance(adb, first_frame=None, expected_yarn=None):
    TRANSITION_HISTORY.clear()
    deadline = time.monotonic() + 120
    began = time.monotonic()
    previous = None
    knitting_seen = False
    next_requested = False
    while time.monotonic() < deadline:
        if first_frame is not None:
            image, first_frame = first_frame, None
        else:
            image = screenshot(adb)
        if next_requested:
            if board_visible(image):
                print(f'Transition +{time.monotonic() - began:.1f}s: board opened', flush=True)
                return
        action = None
        point = None
        if match_ui(image, 'failure'):
            raise HeartLost('Game reports no hearts remaining', image)
        if match_ui(image, 'reward') or match_ui(image, 'collection'):
            action, point = 'dismiss reward', (720, 2500)
        elif match_ui(image, 'knit'):
            action = 'knit'
            knitting_seen = True
        else:
            for name in ('confirm', 'close', 'next'):
                point = match_ui(image, name)
                if point:
                    action = name
                    break
        if action is None and knitting_seen:
            if board_visible(image):
                # An early next-button tap already opened the board.
                print(f'Transition +{time.monotonic() - began:.1f}s: board opened', flush=True)
                return
            # The next button occupies this position on the collection scene.
            # Trying it while its entrance animation runs avoids waiting for
            # a subsequent screenshot once the control accepts input.
            action, point = 'probe next', (1010, 2600)
        if action != previous:
            print(f'Transition +{time.monotonic() - began:.1f}s: {action or "waiting"}' + (f' at {point}' if point else ''), flush=True)
            previous = action
        if action:
            # Keep recent already-acquired frames in memory. They are only
            # written out if this transition or the next board fails.
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                record = {'action': action, 'point': (720, 2550) if action == 'knit' else point,
                          'elapsed_seconds': round(time.monotonic() - began, 3)}
                if action == 'close':
                    record['additional_point_after_0.25s'] = (720, 2295)
                TRANSITION_HISTORY.append((record,
                                           encoded.tobytes()))
        if action == 'knit':
            scale = image.shape[1] / 1440
            x, y = round(720 * scale), round(2550 * scale)
            duration = knit_duration(image, expected_yarn)
            expected_yarn = None
            if duration == 0:
                # The game may keep the knitting overlay open at zero yarn
                # until the button is held and released once more.
                duration = 2000
            if duration:
                print(f'Knitting press: {duration / 1000:.2f}s', flush=True)
                subprocess.run(adb + ['shell', 'cmd', 'input', 'swipe', str(x), str(y), str(x), str(y), str(duration)], check=True)
        elif point:
            if action == 'close':
                # The leaderboard covers the victory dialog. Its close button
                # and the dialog's confirm button never overlap; try confirm
                # as the close animation runs instead of waiting for a frame.
                scale = image.shape[1] / 1440
                x, y = (round(value * scale) for value in point)
                cx, cy = round(720 * scale), round(2295 * scale)
                commands = f'cmd input tap {x} {y}; sleep 0.25; cmd input tap {cx} {cy}'
                subprocess.run(adb + ['shell', commands], check=True)
            else:
                tap(adb, image, point)
            if action in ('next', 'probe next'):
                next_requested = True
        # Screenshot acquisition itself provides a short polling interval.
    raise RuntimeError('Timed out waiting for next-level button')


class RunFailure(Exception):
    """An observed failure, with the screen that revealed it."""

    def __init__(self, message, frame=None):
        super().__init__(message)
        self.frame = frame


class HeartLost(RunFailure):
    """A loss was detected; stop without replaying."""


class ArrowNotCleared(RunFailure):
    """An attempted tap did not clear its arrow."""


def count_hearts(frame):
    """Count red hearts, including when the victory overlay dims the screen."""
    frame = cv2.resize(frame, (1440, round(frame.shape[0] * 1440 / frame.shape[1])))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    samples = [hsv[300:345, x + 15:x + 65] for x in (568, 675, 782)]
    if any(sample.size == 0 or np.median(sample[:, :, 2]) <= 3 for sample in samples):
        return None
    return int(sum(np.count_nonzero((sample[:, :, 0] < 10) &
                                   (sample[:, :, 1] > 80) &
                                   (sample[:, :, 2] > 2)) > 400
                   for sample in samples))


def save_failure(evidence, frame, reason, root=None):
    """Persist observations, not an inferred cause, for offline diagnosis."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory = (Path(root) if root is not None else Path(__file__).resolve().parent / 'failures') / stamp
    directory.mkdir(parents=True, exist_ok=False)
    before = evidence.get('before')
    if before is not None:
        cv2.imwrite(str(directory / 'before.png'), before)
    if frame is not None:
        cv2.imwrite(str(directory / 'after.png'), frame)
    history = []
    for index, (metadata, image) in enumerate(TRANSITION_HISTORY):
        name = f'transition-{index:02d}.jpg'
        (directory / name).write_bytes(image)
        history.append({**metadata, 'image': name})
    details = {key: value for key, value in evidence.items() if key != 'before'}
    if history:
        details['transition_history'] = history
    details['observation'] = reason
    (directory / 'plan.json').write_text(json.dumps(details, ensure_ascii=False, indent=2) + '\n')
    return directory


def stop_with_evidence(adb, evidence, exc):
    frame = getattr(exc, 'frame', None)
    if frame is None:
        try:
            frame = screenshot(adb)
        except Exception:
            pass
    evidence['traceback'] = traceback.format_exc()
    directory = save_failure(evidence, frame, str(exc))
    raise SystemExit(f'{exc}; stopped without replay. Evidence: {directory}') from exc


def restore_color_theme(adb, already_open=False):
    """Refresh the color theme after an animated theme was unlocked."""
    frame = screenshot(adb)
    if not already_open:
        small = cv2.resize(frame, (360, round(frame.shape[0] * 360 / frame.shape[1])))
        if np.min(small[170, 10]) < 240:
            raise RuntimeError('Cannot restore theme: game board is not visible')
        tap(adb, frame, (370, 240))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        frame = screenshot(adb)
        if match_ui(frame, 'theme_menu', region=(450, 550, 1000, 850)):
            break
    else:
        raise RuntimeError('Cannot restore theme: theme menu not found')
    color_region = (550, 1100, 900, 1400)
    color = match_ui(frame, 'color_select', region=color_region)
    if not color:
        # The menu can claim color is selected while the newly unlocked theme
        # remains rendered. Switching away and back forces a refresh.
        alternate = match_ui(frame, 'color_select', region=(950, 1100, 1220, 1400))
        if not alternate:
            raise RuntimeError('Cannot refresh theme: alternate theme not found')
        tap(adb, frame, alternate)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            frame = screenshot(adb)
            color = match_ui(frame, 'color_select', region=color_region)
            if color:
                break
    if not color:
        raise RuntimeError('Cannot restore theme: color option not found')
    tap(adb, frame, color)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        frame = screenshot(adb)
        if match_ui(frame, 'color_selected', region=color_region):
            tap(adb, frame, (1290, 740))
            return
        small = cv2.resize(frame, (360, round(frame.shape[0] * 360 / frame.shape[1])))
        if np.min(small[170, 10]) >= 240:
            # Selecting a theme can close the menu automatically.
            return
    raise RuntimeError('Cannot confirm color theme selection')


def solve_one(adb, args, evidence=None):
    if evidence is None:
        evidence = {}
    planning_started = time.monotonic()
    phase_seconds = {'board': 0.0, 'recognition': 0.0, 'theme': 0.0, 'dependencies': 0.0}
    evidence['phase_seconds'] = phase_seconds

    def timed_call(phase, function, *positional, **keyword):
        started = time.monotonic()
        try:
            return function(*positional, **keyword)
        finally:
            phase_seconds[phase] += time.monotonic() - started

    try:
        image = timed_call('board', stable_board, adb, timeout=8, allow_ui=args.play)
        evidence['before'] = image
        arrows = timed_call('recognition', recognize, image)
    except TexturedTheme:
        if not args.play:
            raise
        print('Textured theme detected; refreshing the color theme', flush=True)
        timed_call('theme', restore_color_theme, adb)
        image = timed_call('board', stable_board, adb, allow_ui=True)
        evidence['before'] = image
        arrows = timed_call('recognition', recognize, image)
    if not arrows:
        raise ValueError('No arrows found')
    blockers = timed_call('dependencies', blockers_for, arrows)
    waves = timed_call('dependencies', plan_waves, arrows, args.batch_size, blockers)
    order = [i for wave in waves for i in wave]
    step_numbers = {arrow_index: step_number
                    for step_number, arrow_index in enumerate(order, 1)}
    scale = image.shape[1] / 1440
    initial_count = count_hearts(image)
    if initial_count is None:
        raise HeartLost('Cannot start: heart count is not readable', image)
    if initial_count < 3:
        raise HeartLost(f'Cannot start: board already has only {initial_count} hearts', image)
    wave_numbers = {arrow_index: wave_number
                    for wave_number, wave in enumerate(waves, 1)
                    for arrow_index in wave}
    evidence.update(before=image, initial_hearts=initial_count, interval=args.interval,
                    verify=args.verify, monitor=args.monitor,
                    screenshot_size=[int(image.shape[1]), int(image.shape[0])],
                    moves=[{'x': round(arrows[i].x * scale), 'y': round(arrows[i].y * scale),
                            'direction': ('left', 'down', 'right', 'up')[arrows[i].direction],
                            'wave': wave_numbers[i], 'score': float(arrows[i].score),
                            'blocked_by_steps': sorted(step_numbers[j] for j in blockers[i])}
                           for i in order], submitted_taps=0, phase_seconds=phase_seconds)
    print(f'{len(arrows)} arrows in {len(waves)} safe batches; capture and plan {time.monotonic() - planning_started:.1f}s:', flush=True)
    print('Planning stages: ' + ', '.join(f'{name}={seconds:.2f}s' for name, seconds in phase_seconds.items()), flush=True)
    for n, i in enumerate(order, 1):
        a = arrows[i]
        print(f'{n}: ({round(a.x*scale)}, {round(a.y*scale)}) {"left down right up".split()[a.direction]} score={a.score:.2f}', flush=True)
    if not args.play:
        return
    # Android executes these in order; no animation wait is needed.
    def heart_pixels(frame):
        frame = cv2.resize(frame, (1440, round(frame.shape[0] * 1440 / frame.shape[1])))
        hsv = cv2.cvtColor(frame[275:370, 550:870], cv2.COLOR_BGR2HSV)
        return np.count_nonzero((hsv[:, :, 0] < 10) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 100))

    initial_hearts = heart_pixels(image)
    input_started = time.monotonic()
    start = 0
    # Screenshots run concurrently with input: they do not create a pause
    # between waves, but can stop the next wave after a collision.
    stop_monitor = threading.Event()
    problems = []
    def monitor():
        while not stop_monitor.is_set():
            try:
                frame = screenshot(adb)
                if stop_monitor.is_set():
                    return
                if match_ui(frame, 'failure'):
                    evidence['detection_inflight_taps'] = evidence.get('inflight_taps')
                    evidence['submitted_taps_at_detection'] = evidence['submitted_taps']
                    problems.append(HeartLost('Game reports no hearts remaining', frame))
                    return
                reduced = cv2.resize(frame, (360, round(frame.shape[0] * 360 / frame.shape[1])))
                if np.min(reduced[170, 10]) >= 240 and heart_pixels(frame) < initial_hearts * .85:
                    evidence['detection_inflight_taps'] = evidence.get('inflight_taps')
                    evidence['submitted_taps_at_detection'] = evidence['submitted_taps']
                    problems.append(HeartLost('Heart count fell during input', frame))
                    return
            except Exception as exc:
                problems.append(RuntimeError(f'Background monitor failed: {exc}'))
                return

    watcher = None
    if not args.verify and args.monitor:
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
    try:
        execution_batches = waves if args.verify or args.monitor else [order]
        for batch in execution_batches:
            if problems:
                raise problems[0]
            evidence['inflight_taps'] = [start + 1, start + len(batch)]
            commands = []
            for offset, i in enumerate(batch):
                commands.append(f'input tap {round(arrows[i].x*scale)} {round(arrows[i].y*scale)}')
                if start + offset + 1 < len(order):
                    commands.append(f'sleep {args.interval}')
            commands = '; '.join(commands)
            subprocess.run(adb + ['shell', commands], check=True)
            start += len(batch)
            evidence['submitted_taps'] = start
            evidence.pop('inflight_taps', None)
            if problems:
                raise problems[0]
            if start == len(order):
                break
            if not args.verify:
                continue
            frame = screenshot(adb)
            if heart_pixels(frame) < initial_hearts * .85:
                raise HeartLost(f'Heart count fell after tap {start}', frame)
            frame = cv2.resize(frame, (1440, round(frame.shape[0] * 1440 / frame.shape[1])))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            colored = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 100)
            def uncleared(pixels):
                return [i for i in batch if np.count_nonzero(pixels & arrows[i].mask) / np.count_nonzero(arrows[i].mask) > .4]
            pending = uncleared(colored)
            if pending:
                # Long lines may still be moving through their original position.
                time.sleep(.6)
                frame = screenshot(adb)
                if heart_pixels(frame) < initial_hearts * .85:
                    raise HeartLost(f'Heart count fell after tap {start}', frame)
                frame = cv2.resize(frame, (1440, round(frame.shape[0] * 1440 / frame.shape[1])))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                colored = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 100)
                if uncleared(colored):
                    raise ArrowNotCleared(f'Arrow still present after tap batch ending at {start}', frame)
    finally:
        stop_monitor.set()
        if watcher:
            watcher.join(timeout=2)
    if problems:
        raise problems[0]
    print(f'All taps sent in {time.monotonic() - input_started:.1f}s; waiting for victory screen', flush=True)
    return len(arrows)


def wait_for_victory(adb, initial_hearts):
    # Some levels spend more than ten seconds in the game's post-clear
    # animation before the normal victory controls appear. Polling still
    # returns immediately for the common fast case; the longer deadline
    # only prevents a false timeout on slow animations.
    deadline = time.monotonic() + 30
    low_heart_count = 0
    while time.monotonic() < deadline:
        frame = screenshot(adb)
        if match_ui(frame, 'failure'):
            raise HeartLost('Game reports no hearts remaining', frame)
        states = [name for name in ('victory', 'next', 'close', 'confirm', 'knit', 'collection')
                  if match_ui(frame, name)]
        reduced = cv2.resize(frame, (360, round(frame.shape[0] * 360 / frame.shape[1])))
        board_visible = np.min(reduced[170, 10]) >= 240
        check_hearts = board_visible or any(name in states for name in ('victory', 'close', 'confirm'))
        if check_hearts:
            remaining = count_hearts(frame)
            if remaining is not None and remaining < initial_hearts:
                # Victory confetti can briefly cover a heart. Confirm a low
                # count in several frames before reporting a loss; no taps
                # are issued while waiting for the animation to settle.
                low_heart_count += 1
                if low_heart_count >= 3:
                    raise HeartLost(f'Heart count fell from {initial_hearts} to {remaining}', frame)
                time.sleep(.3)
                continue
            low_heart_count = 0
            if states and remaining is None:
                # Do not accept an obscured victory frame as evidence that no
                # life was lost; wait for a readable post-game frame.
                time.sleep(.1)
                continue
        if states:
            return frame
        time.sleep(.1)
    raise RunFailure('Victory was not detected after all taps', frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--levels', type=int, default=0, help='number of levels to solve; 0 runs continuously (default)')
    parser.add_argument('--interval', type=float, default=.08, help='seconds between taps')
    parser.add_argument('--batch-size', type=int, default=6, help='maximum independent taps per batch when --verify is enabled')
    parser.add_argument('--verify', action='store_true', help='check screenshots between independent batches')
    parser.add_argument('--monitor', action='store_true', help='monitor hearts asynchronously without waiting between batches')
    parser.add_argument('--advance-first', action='store_true', help='start from the victory screen')
    args = parser.parse_args()
    if args.levels < 0 or not math.isfinite(args.interval) or args.interval < 0 or args.batch_size < 1:
        parser.error('--levels and --interval must be non-negative and finite; --batch-size must be positive')
    adb = ['adb', '-s', args.device]
    try:
        if not args.advance_first and args.play:
            initial = screenshot(adb)
            if match_ui(initial, 'theme_menu', region=(450, 550, 1000, 850)):
                restore_color_theme(adb, already_open=True)
            elif match_ui(initial, 'failure'):
                raise HeartLost('Game was already on the failure screen', initial)
            else:
                states = [name for name in ('victory', 'next', 'close', 'confirm', 'knit', 'collection')
                          if match_ui(initial, name)]
                if any(name in states for name in ('victory', 'close', 'confirm')):
                    hearts = count_hearts(initial)
                    if hearts is None or hearts < 3:
                        raise HeartLost(f'Cannot advance: previous result has {hearts} readable hearts', initial)
                if states:
                    args.advance_first = True
        if args.advance_first:
            if not args.play:
                parser.error('--advance-first requires --play')
            advance(adb)
    except Exception as exc:
        stop_with_evidence(adb, {}, exc)
    import itertools
    for level in (range(args.levels) if args.levels else itertools.count()):
        print(f'Level {level + 1}: planning', flush=True)
        evidence = {'run_level_index': level + 1}
        try:
            arrow_count = solve_one(adb, args, evidence)
            if not args.play:
                return
            victory_started = time.monotonic()
            victory_frame = wait_for_victory(adb, evidence['initial_hearts'])
            print(f'Victory confirmed after {time.monotonic() - victory_started:.1f}s', flush=True)
            if not args.levels or level + 1 < args.levels:
                transition_started = time.monotonic()
                advance(adb, first_frame=victory_frame, expected_yarn=arrow_count)
                print(f'Next level opened in {time.monotonic() - transition_started:.1f}s', flush=True)
        except Exception as exc:
            stop_with_evidence(adb, evidence, exc)


if __name__ == '__main__':
    main()
