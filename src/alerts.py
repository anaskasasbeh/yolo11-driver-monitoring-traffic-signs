"""Turn per-frame detections into spoken alerts.

Design:

1. Detections repeat on every frame and sometimes flicker. ``AlertManager``
   applies two rules per event: it must be seen continuously for
   ``min_duration`` seconds, and the same alert is not repeated within
   ``cooldown`` seconds.
2. ``pyttsx3.runAndWait()`` blocks until speech ends. ``Speaker`` runs
   text-to-speech on a background thread so the video never pauses.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertRule:
    """When and what to say for one detected event.

    Attributes:
        message: Sentence spoken to the driver.
        min_duration: Seconds the event must be seen continuously before alerting.
        cooldown: Minimum seconds between two alerts for the same event.
    """

    message: str
    min_duration: float = 0.0
    cooldown: float = 10.0


class AlertManager:
    """Decide which alerts to fire, given the events seen in each frame.

    Call :meth:`update` once per frame, even when nothing was detected.

    Args:
        rules: Mapping from event name (a class name or derived event) to its rule.
        grace: Short gap in seconds that does not break a "continuous" sighting.
            Detectors often miss a single frame; without this, one missed frame
            would restart the ``min_duration`` timer. An event seen in the
            previous frame always counts as continuous, so slow frame rates work too.
    """

    def __init__(self, rules: dict[str, AlertRule], grace: float = 0.3) -> None:
        self.rules = rules
        self.grace = grace
        self._first_seen: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._last_alert: dict[str, float] = {}
        self._previous_events: set[str] = set()

    def update(self, events: set[str], now: float | None = None) -> list[str]:
        """Register the events seen in the current frame.

        Args:
            events: Names of everything detected (or derived) in this frame.
            now: Current time in seconds; defaults to ``time.monotonic()``.

        Returns:
            Messages to speak now (usually empty).
        """
        now = time.monotonic() if now is None else now
        messages: list[str] = []

        for event in sorted(events):
            rule = self.rules.get(event)
            if rule is None:
                continue

            last_seen = self._last_seen.get(event)
            continuous = event in self._previous_events or (last_seen is not None and now - last_seen <= self.grace)
            if not continuous:
                self._first_seen[event] = now  # a new continuous sighting starts
            self._last_seen[event] = now

            seen_for = now - self._first_seen[event]
            last_alert = self._last_alert.get(event)
            cooled_down = last_alert is None or now - last_alert >= rule.cooldown
            if seen_for >= rule.min_duration and cooled_down:
                self._last_alert[event] = now
                messages.append(rule.message)

        self._previous_events = set(events)
        return messages


class Speaker:
    """Speak messages with pyttsx3 on a background thread.

    The video loop calls :meth:`say` and continues immediately. If text-to-speech
    is unavailable (no speech engine installed, ``--mute``), messages are only
    logged, so detection keeps working.

    Args:
        enabled: Set to False to mute all speech.
        max_pending: Messages waiting beyond this are dropped; a stale warning
            spoken several seconds late is worse than none.
    """

    def __init__(self, enabled: bool = True, max_pending: int = 2) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_pending)
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        """Queue a message without blocking the caller."""
        logger.info("ALERT: %s", text)
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.debug("Speech queue full, dropped: %s", text)

    def close(self, timeout: float = 3.0) -> None:
        """Stop the speech thread after the current message."""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass  # the thread is busy speaking; it is a daemon and ends with the program
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        com_initialized = _init_windows_com()
        try:
            import pyttsx3
        except ImportError:
            logger.warning("pyttsx3 is not installed; alerts will only be logged.")
            self.enabled = False
            return

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                # A fresh engine per message avoids a known pyttsx3 issue where
                # runAndWait() stops speaking after the first call in a loop.
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as exc:  # noqa: BLE001 - any engine failure means "no voice"
                logger.warning("Text-to-speech failed (%s); alerts will only be logged.", exc)
                self.enabled = False
                break

        if com_initialized:
            import pythoncom

            pythoncom.CoUninitialize()


def _init_windows_com() -> bool:
    """Initialise COM for this thread on Windows (needed by the SAPI5 voice)."""
    if sys.platform != "win32":
        return False
    try:
        import pythoncom  # installed with pywin32, a pyttsx3 dependency on Windows
    except ImportError:
        return False
    pythoncom.CoInitialize()
    return True
