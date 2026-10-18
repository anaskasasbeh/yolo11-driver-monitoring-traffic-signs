"""Unit tests for the alert timing rules (no camera or model needed).

Run with:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alerts import AlertManager, AlertRule  # noqa: E402
from config import DMS, TSD, derive_dms_events  # noqa: E402

FRAME = 0.1  # seconds between frames, i.e. 10 FPS


def feed(manager: AlertManager, frames: list[set[str]], start: float = 0.0) -> list[tuple[float, str]]:
    """Send one event set per frame and collect (time, message) for every alert."""
    fired = []
    for i, events in enumerate(frames):
        now = start + i * FRAME
        fired += [(round(now, 2), m) for m in manager.update(events, now=now)]
    return fired


class AlertManagerTest(unittest.TestCase):
    def test_short_blink_does_not_alert(self):
        manager = AlertManager({"Closed Eye": AlertRule("Wake up", min_duration=1.0)})
        frames = [{"Closed Eye"}] * 3 + [{"Open Eye"}] * 20  # eyes closed for 0.3 s
        self.assertEqual(feed(manager, frames), [])

    def test_long_eye_closure_alerts_once_after_min_duration(self):
        manager = AlertManager({"Closed Eye": AlertRule("Wake up", min_duration=1.0, cooldown=5.0)})
        fired = feed(manager, [{"Closed Eye"}] * 20)  # closed for 2 s
        self.assertEqual(fired, [(1.0, "Wake up")])

    def test_cooldown_allows_repeat_after_it_expires(self):
        manager = AlertManager({"Phone": AlertRule("Phone down", min_duration=0.0, cooldown=1.0)})
        fired = feed(manager, [{"Phone"}] * 25)  # 2.5 s of phone use
        self.assertEqual([t for t, _ in fired], [0.0, 1.0, 2.0])

    def test_single_missed_frame_does_not_reset_timer(self):
        manager = AlertManager({"Closed Eye": AlertRule("Wake up", min_duration=1.0)}, grace=0.3)
        frames = [{"Closed Eye"}] * 6 + [set()] + [{"Closed Eye"}] * 6  # detector missed one frame
        self.assertEqual(feed(manager, frames), [(1.0, "Wake up")])  # timer kept running from 0.0

    def test_long_gap_resets_timer(self):
        manager = AlertManager({"Closed Eye": AlertRule("Wake up", min_duration=1.0)}, grace=0.3)
        frames = [{"Closed Eye"}] * 6 + [set()] * 10 + [{"Closed Eye"}] * 6
        self.assertEqual(feed(manager, frames), [])

    def test_slow_frame_rate_still_counts_as_continuous(self):
        manager = AlertManager({"Closed Eye": AlertRule("Wake up", min_duration=1.0)}, grace=0.3)
        fired = [m for i in range(4) for m in manager.update({"Closed Eye"}, now=i * 0.5)]  # 2 FPS
        self.assertEqual(fired, ["Wake up"])

    def test_events_without_rules_are_ignored(self):
        manager = AlertManager({"Stop": AlertRule("Stop")})
        self.assertEqual(manager.update({"Green Light"}, now=0.0), [])


class TaskConfigTest(unittest.TestCase):
    def test_missing_seatbelt_is_derived_only_when_driver_is_visible(self):
        self.assertIn("No Seatbelt", derive_dms_events({"Open Eye"}))
        self.assertNotIn("No Seatbelt", derive_dms_events({"Open Eye", "Seatbelt"}))
        self.assertNotIn("No Seatbelt", derive_dms_events(set()))  # empty seat: no warning

    def test_every_alert_rule_matches_a_class_or_derived_event(self):
        for task, derived in ((DMS, {"No Seatbelt"}), (TSD, set())):
            known = set(task.class_names) | derived
            self.assertTrue(set(task.alert_rules) <= known, task.name)

    def test_tsd_has_rules_for_all_speed_limits_but_not_green_light(self):
        speed_classes = {c for c in TSD.class_names if c.startswith("Speed Limit")}
        self.assertTrue(speed_classes <= set(TSD.alert_rules))
        self.assertNotIn("Green Light", TSD.alert_rules)


if __name__ == "__main__":
    unittest.main()
