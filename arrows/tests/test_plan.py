import unittest
import tempfile
import json
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from arrows.solve import (Arrow, TEMPLATE, TRANSITION_HISTORY, HeartLost, RunFailure, blockers_for, board_visible, count_hearts, match_ui,
                          crosses_tap, flat_palette_fraction, main, plan_waves, recognize, save_failure, solve_one, stable_board, tap, wait_for_victory)


def arrow(x, y, direction, pixels):
    mask = np.zeros((80, 80), dtype=bool)
    for px, py in pixels:
        mask[py, px] = True
    return Arrow(mask, x, y, direction, 1.0, 2)


def heart_frame(count, dim=False):
    frame = np.full((3200, 1440, 3), 255, dtype=np.uint8)
    for i, x in enumerate((568, 675, 782)):
        frame[300:345, x + 15:x + 65] = (20, 20, 220) if i < count else (150, 100, 100)
    if dim:
        frame = (frame.astype(np.float32) * .05).astype(np.uint8)
    return frame


class WavePlanningTests(unittest.TestCase):
    def test_blocked_arrow_is_not_in_same_wave_as_blocker(self):
        arrows = [arrow(15, 20, 2, [(15, 20)]),
                  arrow(40, 20, 3, [(40, 20)]),
                  arrow(15, 55, 0, [(15, 55)])]
        waves = plan_waves(arrows, 6)
        self.assertIn(1, blockers_for(arrows)[0])
        self.assertLess(next(i for i, wave in enumerate(waves) if 1 in wave),
                        next(i for i, wave in enumerate(waves) if 0 in wave))
        remaining = set(range(len(arrows)))
        for wave in waves:
            self.assertTrue(all(not blockers_for(arrows)[i] & remaining for i in wave))
            remaining.difference_update(wave)

    def test_moving_head_cannot_cross_a_later_tap(self):
        moving = arrow(20, 20, 2, [(20, 20)])
        target = arrow(40, 20, 3, [(40, 20)])
        self.assertTrue(crosses_tap(moving, target))
        self.assertFalse(crosses_tap(target, moving))

    def test_cycle_is_rejected_before_any_tap(self):
        arrows = [arrow(20, 20, 2, [(20, 20)]), arrow(40, 20, 0, [(40, 20)])]
        with self.assertRaisesRegex(ValueError, 'No collision-free move'):
            plan_waves(arrows, 6)


class RecognitionTests(unittest.TestCase):
    def test_failure_persists_recent_transition_frames(self):
        TRANSITION_HISTORY.clear()
        self.addCleanup(TRANSITION_HISTORY.clear)
        TRANSITION_HISTORY.append(({'action': 'next', 'point': (1000, 2600)}, b'jpeg-data'))
        with tempfile.TemporaryDirectory() as root:
            directory = save_failure({}, None, 'test', root=root)
            data = json.loads((directory / 'plan.json').read_text())
            self.assertEqual(data['transition_history'][0]['point'], [1000, 2600])
            self.assertEqual((directory / 'transition-00.jpg').read_bytes(), b'jpeg-data')

    def test_white_share_page_is_not_board(self):
        frame = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        self.assertFalse(board_visible(frame))
        frame = heart_frame(3)
        frame[1000:1200, 400:800] = (40, 180, 240)
        self.assertTrue(board_visible(frame))

    @patch('arrows.solve.ui_template')
    def test_next_button_requires_green_not_orange(self, ui_template):
        rng = np.random.default_rng(9)
        pattern = rng.integers(0, 35, (20, 50, 1), dtype=np.uint8)
        template = np.concatenate((pattern + 30, pattern + 150, pattern + 30), axis=2)
        ui_template.return_value = template
        frame = np.zeros((3200, 1440, 3), dtype=np.uint8)
        orange = np.concatenate((pattern + 30, pattern + 110, pattern + 190), axis=2)
        frame[2000:2080, 1000:1200] = cv2.resize(orange, (200, 80), interpolation=cv2.INTER_NEAREST)
        self.assertIsNone(match_ui(frame, 'next'))
        frame[2000:2080, 1000:1200] = cv2.resize(template, (200, 80), interpolation=cv2.INTER_NEAREST)
        self.assertIsNotNone(match_ui(frame, 'next'))

    def test_arrow_crossing_previous_top_crop_is_recognized(self):
        image = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        template = cv2.resize(TEMPLATE, (37, 37), interpolation=cv2.INTER_NEAREST)
        image[620:657, 500:537][template > 0] = (200, 40, 240)
        arrows = recognize(image)
        self.assertEqual(len(arrows), 1)
        self.assertEqual(arrows[0].direction, 0)

    def test_palette_fraction_matches_reference_count(self):
        rng = np.random.default_rng(17)
        board = rng.integers(0, 256, size=(50, 70, 3), dtype=np.uint8)
        mask = rng.random((50, 70)) > .35
        _, counts = np.unique(board[mask] // 8, axis=0, return_counts=True)
        expected = np.sort(counts)[-8:].sum() / mask.sum()
        self.assertEqual(flat_palette_fraction(board, mask), expected)
        self.assertEqual(flat_palette_fraction(board, np.zeros_like(mask)), 0.0)

    def test_short_arrow_under_old_area_threshold_is_not_discarded(self):
        image = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        template = cv2.resize(TEMPLATE, (25, 25), interpolation=cv2.INTER_NEAREST)
        patch_image = image[1000:1025, 500:525]
        patch_image[template > 0] = (200, 40, 240)
        arrows = recognize(image)
        self.assertEqual(len(arrows), 1)
        self.assertLess(np.count_nonzero(arrows[0].mask), 500)
        self.assertEqual(arrows[0].direction, 0)

    def test_small_unmodeled_fragment_is_not_treated_as_empty(self):
        image = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        image[1000:1003, 500:503] = (200, 40, 240)
        with self.assertRaisesRegex(ValueError, 'small colored component.*area 9'):
            recognize(image)

    def test_parallel_recognition_matches_single_worker(self):
        image = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        template = cv2.resize(TEMPLATE, (25, 25), interpolation=cv2.INTER_NEAREST)
        image[1000:1025, 500:525][template > 0] = (200, 40, 240)
        image[1600:1625, 900:925][np.rot90(template) > 0] = (240, 200, 40)
        serial = recognize(image, workers=1)
        parallel = recognize(image, workers=4)
        self.assertEqual(len(serial), 2)
        self.assertEqual(len(parallel), len(serial))
        for a, b in zip(serial, parallel):
            self.assertEqual((a.x, a.y, a.direction, a.radius, a.score),
                             (b.x, b.y, b.direction, b.radius, b.score))
            np.testing.assert_array_equal(a.mask, b.mask)
        self.assertEqual(plan_waves(serial, 6), plan_waves(parallel, 6))


class ExecutionTests(unittest.TestCase):
    @patch('arrows.solve.subprocess.run')
    def test_ui_tap_uses_same_input_service_with_scaled_coordinates(self, run):
        image = np.zeros((1600, 720, 3), dtype=np.uint8)
        tap(['adb', '-s', 'device'], image, (720, 2000))
        run.assert_called_once_with(['adb', '-s', 'device', 'shell', 'cmd', 'input', 'tap', '360', '1000'], check=True)

    @patch('arrows.solve.subprocess.run')
    @patch('arrows.solve.count_hearts', return_value=2)
    @patch('arrows.solve.recognize')
    @patch('arrows.solve.stable_board')
    def test_board_with_missing_heart_stops_before_any_tap(self, stable_board, recognize, heart_count, run):
        stable_board.return_value = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        recognize.return_value = [arrow(15, 20, 2, [(15, 20)])]
        args = SimpleNamespace(batch_size=6, play=True, verify=False, interval=.08, monitor=False, min_hearts=3)
        with self.assertRaisesRegex(HeartLost, 'already has only 2'):
            solve_one(['adb'], args)
        run.assert_not_called()

    @patch('arrows.solve.subprocess.run')
    @patch('arrows.solve.count_hearts', return_value=2)
    @patch('arrows.solve.recognize')
    @patch('arrows.solve.stable_board')
    def test_lower_minimum_allows_two_hearts(self, stable_board, recognize, heart_count, run):
        stable_board.return_value = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        recognize.return_value = [arrow(15, 20, 2, [(15, 20)])]
        args = SimpleNamespace(batch_size=6, play=True, verify=False, interval=.08, monitor=False, min_hearts=2)
        evidence = {}
        self.assertEqual(solve_one(['adb'], args, evidence), 1)
        self.assertEqual(evidence['initial_hearts'], 2)
        self.assertEqual(evidence['min_hearts'], 2)
        run.assert_called_once()

    @patch('time.sleep')
    @patch('arrows.solve.match_ui', return_value=None)
    @patch('arrows.solve.screenshot', return_value=heart_frame(1))
    def test_loss_after_starting_with_two_hearts_stops(self, screenshot, match_ui, sleep):
        with self.assertRaisesRegex(HeartLost, 'from 2 to 1'):
            wait_for_victory(['adb'], 2)

    @patch('arrows.solve.count_hearts', return_value=3)
    @patch('arrows.solve.subprocess.run')
    @patch('arrows.solve.screenshot', side_effect=AssertionError('unexpected intermediate screenshot'))
    @patch('arrows.solve.plan_waves', return_value=[[0], [1]])
    @patch('arrows.solve.recognize')
    @patch('arrows.solve.stable_board')
    def test_fast_mode_sends_entire_plan_without_intermediate_screenshot(
            self, stable_board, recognize, waves, screenshot, run, heart_count):
        stable_board.return_value = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        recognize.return_value = [arrow(15, 20, 2, [(15, 20)]),
                                  arrow(40, 20, 3, [(40, 20)])]
        args = SimpleNamespace(batch_size=6, play=True, verify=False, interval=.08, monitor=False, min_hearts=3)
        self.assertEqual(solve_one(['adb'], args), 2)
        self.assertEqual(run.call_count, 1)
        commands = [c.args[0][-1] for c in run.call_args_list]
        self.assertEqual(sum(command.count('input tap') for command in commands), 2)
        self.assertEqual(commands[0].count('sleep 0.08'), 1)
        self.assertFalse(commands[0].endswith('sleep 0.08'))
        screenshot.assert_not_called()

    @patch('arrows.solve.count_hearts', return_value=3)
    @patch('arrows.solve.subprocess.run')
    @patch('arrows.solve.recognize')
    @patch('arrows.solve.stable_board')
    def test_failure_evidence_records_original_blockers(self, stable_board, recognize, run, heart_count):
        stable_board.return_value = np.full((3200, 1440, 3), 255, dtype=np.uint8)
        recognize.return_value = [arrow(15, 20, 2, [(15, 20)]),
                                  arrow(40, 20, 3, [(40, 20)])]
        args = SimpleNamespace(batch_size=6, play=True, verify=False, interval=.08, monitor=False, min_hearts=3)
        evidence = {}
        solve_one(['adb'], args, evidence)
        moves = evidence['moves']
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0]['blocked_by_steps'], [])
        self.assertEqual(moves[1]['blocked_by_steps'], [1])
        self.assertEqual(evidence['submitted_taps'], 2)

    @patch('sys.argv', ['solve.py', '--device', 'test', '--play', '--interval', 'nan'])
    @patch('arrows.solve.screenshot')
    def test_nonfinite_interval_is_rejected_before_device_access(self, screenshot):
        with self.assertRaises(SystemExit):
            main()
        screenshot.assert_not_called()

    @patch('arrows.solve.restore_color_theme')
    @patch('arrows.solve.recognize', side_effect=ValueError('unrecognized component'))
    @patch('arrows.solve.stable_board', return_value=np.zeros((3200, 1440, 3), dtype=np.uint8))
    def test_unrecognized_shape_is_not_hidden_by_theme_retry(self, stable_board, recognize, restore):
        args = SimpleNamespace(batch_size=6, play=True, verify=False, interval=.08, monitor=False, min_hearts=3)
        evidence = {}
        with self.assertRaisesRegex(ValueError, 'unrecognized component'):
            solve_one(['adb'], args, evidence)
        self.assertIn('before', evidence)
        restore.assert_not_called()


class PreviewTests(unittest.TestCase):
    @patch('arrows.solve.tap')
    @patch('arrows.solve.match_ui', return_value=(1, 1))
    @patch('arrows.solve.screenshot', return_value=np.zeros((3200, 1440, 3), dtype=np.uint8))
    def test_preview_does_not_dismiss_reward(self, screenshot, match_ui, tap):
        with self.assertRaisesRegex(RunFailure, 'read-only'):
            stable_board(['adb'], allow_ui=False)
        tap.assert_not_called()


class FailureTests(unittest.TestCase):
    def test_heart_counter_handles_dimmed_victory_screen(self):
        self.assertEqual(count_hearts(heart_frame(3)), 3)
        self.assertEqual(count_hearts(heart_frame(1)), 1)
        self.assertEqual(count_hearts(heart_frame(3, dim=True)), 3)
        self.assertEqual(count_hearts(heart_frame(1, dim=True)), 1)

    def test_failure_evidence_contains_images_and_plan(self):
        before, after = heart_frame(3), heart_frame(2)
        evidence = {'before': before, 'initial_hearts': count_hearts(before),
                    'moves': [{'x': 10, 'y': 20}], 'submitted_taps': 1}
        with tempfile.TemporaryDirectory() as root:
            directory = save_failure(evidence, after, 'heart count fell', root=root)
            self.assertTrue((directory / 'before.png').is_file())
            self.assertTrue((directory / 'after.png').is_file())
            saved = json.loads((directory / 'plan.json').read_text())
            self.assertEqual(saved['initial_hearts'], 3)
            self.assertEqual(saved['moves'][0], {'x': 10, 'y': 20})
            self.assertEqual(saved['observation'], 'heart count fell')

    @patch('arrows.solve.tap')
    @patch('arrows.solve.match_ui', return_value=None)
    @patch('arrows.solve.screenshot', return_value=heart_frame(2))
    def test_heart_loss_after_input_stops_without_tapping(self, screenshot, match_ui, tap):
        with self.assertRaisesRegex(HeartLost, 'from 3 to 2'):
            wait_for_victory(['adb'], 3)
        tap.assert_not_called()

    @patch('time.sleep')
    @patch('arrows.solve.match_ui')
    @patch('arrows.solve.screenshot', side_effect=[np.zeros((3200, 1440, 3), dtype=np.uint8), heart_frame(3)])
    def test_obscured_victory_waits_for_readable_hearts(self, screenshot, match_ui, sleep):
        match_ui.side_effect = lambda frame, name: (1, 1) if name == 'victory' else None
        result = wait_for_victory(['adb'], 3)
        self.assertEqual(count_hearts(result), 3)
        self.assertEqual(screenshot.call_count, 2)

    @patch('time.sleep')
    @patch('arrows.solve.match_ui')
    @patch('arrows.solve.screenshot', side_effect=[heart_frame(2), heart_frame(3)])
    def test_transient_obscured_heart_is_not_reported_as_loss(self, screenshot, match_ui, sleep):
        match_ui.side_effect = lambda frame, name: (1, 1) if name == 'victory' else None
        result = wait_for_victory(['adb'], 3)
        self.assertEqual(count_hearts(result), 3)
        self.assertEqual(screenshot.call_count, 2)

    @patch('time.sleep')
    @patch('arrows.solve.match_ui')
    @patch('arrows.solve.screenshot')
    @patch('arrows.solve.time.monotonic')
    def test_slow_victory_animation_is_not_a_ten_second_failure(self, clock, screenshot, match_ui, sleep):
        clock.side_effect = [0] + list(range(13))
        pending = np.zeros((3200, 1440, 3), dtype=np.uint8)
        success = heart_frame(3)
        screenshot.side_effect = [pending] * 11 + [success]
        match_ui.side_effect = lambda frame, name: (1, 1) if frame is success and name == 'victory' else None
        self.assertIs(wait_for_victory(['adb'], 3), success)
        self.assertEqual(screenshot.call_count, 12)

    @patch('sys.argv', ['solve.py', '--device', 'test', '--play'])
    @patch('arrows.solve.advance')
    @patch('arrows.solve.save_failure', return_value='/tmp/evidence')
    @patch('arrows.solve.match_ui')
    @patch('arrows.solve.screenshot', return_value=heart_frame(2, dim=True))
    def test_previous_victory_with_missing_heart_is_not_advanced(self, screenshot, match_ui, save, advance):
        match_ui.side_effect = lambda frame, name, **kwargs: (1, 1) if name == 'close' else None
        with self.assertRaisesRegex(SystemExit, 'stopped without replay'):
            main()
        save.assert_called_once()
        advance.assert_not_called()

    @patch('sys.argv', ['solve.py', '--device', 'test', '--play'])
    @patch('arrows.solve.tap')
    @patch('arrows.solve.advance')
    @patch('arrows.solve.save_failure', return_value='/tmp/evidence')
    @patch('arrows.solve.match_ui')
    @patch('arrows.solve.screenshot', return_value=heart_frame(1))
    def test_starting_on_failure_screen_never_taps_replay(self, screenshot, match_ui, save, advance, tap):
        match_ui.side_effect = lambda frame, name, **kwargs: (1, 1) if name == 'failure' else None
        with self.assertRaisesRegex(SystemExit, 'stopped without replay'):
            main()
        save.assert_called_once()
        advance.assert_not_called()
        tap.assert_not_called()

    @patch('sys.argv', ['solve.py', '--device', 'test', '--play'])
    @patch('arrows.solve.advance')
    @patch('arrows.solve.save_failure', return_value='/tmp/evidence')
    @patch('arrows.solve.solve_one', side_effect=HeartLost('collision', heart_frame(1)))
    @patch('arrows.solve.match_ui', return_value=None)
    @patch('arrows.solve.screenshot', return_value=heart_frame(3))
    def test_main_stops_on_loss_without_replay(self, screenshot, match_ui, solve, save, advance):
        with self.assertRaisesRegex(SystemExit, 'stopped without replay'):
            main()
        self.assertEqual(solve.call_count, 1)
        save.assert_called_once()
        advance.assert_not_called()

    @patch('sys.argv', ['solve.py', '--device', 'test', '--play'])
    @patch('arrows.solve.advance')
    @patch('arrows.solve.save_failure', return_value='/tmp/evidence')
    @patch('arrows.solve.wait_for_victory', side_effect=RunFailure('victory missing', heart_frame(3)))
    @patch('arrows.solve.solve_one')
    @patch('arrows.solve.match_ui', return_value=None)
    @patch('arrows.solve.screenshot', return_value=heart_frame(3))
    def test_missing_victory_stops_without_replanning(self, screenshot, match_ui, solve, wait, save, advance):
        def solved(adb, args, evidence):
            evidence['initial_hearts'] = 3
            return 2
        solve.side_effect = solved
        with self.assertRaisesRegex(SystemExit, 'stopped without replay'):
            main()
        self.assertEqual(solve.call_count, 1)
        save.assert_called_once()
        advance.assert_not_called()


if __name__ == '__main__':
    unittest.main()
