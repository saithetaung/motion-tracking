"""Visual theme and motion-state classification for the tracking overlay.

Everything that is a "design decision" (colors, thresholds, opacity) lives
here, so restyling the UI never means touching drawing code.
"""
from dataclasses import dataclass
from enum import Enum


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (B, G, R), which is the channel order OpenCV draws in."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class MotionState(Enum):
    """Each state owns a label and a color; the whole overlay for a person is tinted by it."""
    STILL = ("STILL", "#9AA5B1")
    STANDING = ("STANDING", "#34D399")
    JOGGING = ("JOGGING", "#FBBF24")
    RUNNING = ("RUNNING", "#F43F5E")

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def color(self) -> tuple[int, int, int]:
        return hex_to_bgr(self.value[1])


@dataclass(frozen=True)
class Theme:
    # speed thresholds (km/h) between motion states
    still_kmh: float = 1.0
    walk_kmh: float = 6.5
    jog_kmh: float = 12.0

    # info card
    card_bg: tuple = hex_to_bgr("#0F1218")
    card_alpha: float = 0.74
    text: tuple = hex_to_bgr("#F5F7FA")
    muted: tuple = hex_to_bgr("#AAB2BE")
    shadow: tuple = hex_to_bgr("#05070A")

    # closer / farther indicator next to the distance
    trend_min_mps: float = 0.25
    closer_color: tuple = hex_to_bgr("#FB923C")
    farther_color: tuple = hex_to_bgr("#60A5FA")

    # motion trail
    trail_seconds: float = 1.2

    # HUD
    live_color: tuple = hex_to_bgr("#EF4444")

    def classify(self, speed_kmh: float) -> MotionState:
        if speed_kmh < self.still_kmh:
            return MotionState.STILL
        if speed_kmh < self.walk_kmh:
            return MotionState.STANDING
        if speed_kmh < self.jog_kmh:
            return MotionState.JOGGING
        return MotionState.RUNNING
