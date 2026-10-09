"""Tracking overlay: corner brackets, info card, motion arrow, trail, HUD.

Everything drawn for one person is tinted by their motion state
(still / standing / jogging / running), so you can read "who is moving fast"
from across the room without reading any numbers.

Usage (no pipeline change needed):
    from src.visualization.overlay import draw_tracks
    frame = draw_tracks(frame, tracks)
"""
import math
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from src.schemas import Track
from src.visualization import primitives as P
from src.visualization.theme import MotionState, Theme


class OverlayRenderer:
    """Stateful because trails and FPS need memory between frames."""

    def __init__(self, theme: Theme | None = None, show_trails: bool = True,
                 show_hud: bool = True, show_id_with_name: bool = False):
        self.theme = theme or Theme()
        self.show_trails = show_trails
        self.show_hud = show_hud
        self.show_id_with_name = show_id_with_name

        self._trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=120))
        self._last_ts: float | None = None
        self._fps = 0.0

    # ------------------------------------------------------------------ public
    def draw(self, frame: np.ndarray, tracks: list[Track], now: float | None = None) -> np.ndarray:
        now = time.time() if now is None else now
        scale = min(1.5, max(0.6, frame.shape[0] / 1080))   # UI scales with resolution
        self._tick_fps(now)

        for t in tracks:
            self._draw_track(frame, t, scale, now)
        self._prune_trails(now, {t.track_id for t in tracks})

        if self.show_hud:
            self._draw_hud(frame, len(tracks), scale)
        return frame

    # --------------------------------------------------------------- per track
    def _draw_track(self, frame: np.ndarray, t: Track, s: float, now: float) -> None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in t.bbox)
        x1, x2 = sorted((min(max(x1, 0), W - 1), min(max(x2, 0), W - 1)))
        y1, y2 = sorted((min(max(y1, 0), H - 1), min(max(y2, 0), H - 1)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return

        speed = t.speed_kmh or 0.0
        state = self.theme.classify(speed)
        color = state.color
        feet = ((x1 + x2) // 2, y2)

        self._trail(frame, t.track_id, feet, color, state, s, now)
        self._ground_marker(frame, feet, x2 - x1, color, s)
        self._brackets(frame, x1, y1, x2, y2, color, s)
        self._motion_arrow(frame, t, (x1, y1, x2, y2), speed, color, s)
        self._card(frame, t, state, speed, (x1, y1, x2, y2), s)

    def _trail(self, frame, track_id, feet, color, state, s, now) -> None:
        trail = self._trails[track_id]
        trail.append((now, feet))
        if not self.show_trails or state is MotionState.STILL:
            return
        pts = [p for ts, p in trail if now - ts <= self.theme.trail_seconds]
        for i in range(1, len(pts)):
            k = i / len(pts)                                   # 0 = oldest, 1 = newest
            cv2.line(frame, pts[i - 1], pts[i], color, max(2, int(round((2 + 4 * k) * s))), P.AA)

    def _ground_marker(self, frame, feet, box_w, color, s) -> None:
        rx = int(max(22 * s, box_w * 0.32))
        ry = max(int(6 * s), int(rx * 0.22))
        cv2.ellipse(frame, feet, (rx + 1, ry + 1), 0, 0, 360, self.theme.shadow, max(2, int(3 * s)), P.AA)
        cv2.ellipse(frame, feet, (rx, ry), 0, 0, 360, color, max(1, int(2 * s)), P.AA)

    def _brackets(self, frame, x1, y1, x2, y2, color, s) -> None:
        length = int(np.clip(min(x2 - x1, y2 - y1) * 0.18, 14 * s, 44 * s))
        thick = max(2, int(round(3 * s)))
        P.corner_brackets(frame, x1, y1, x2, y2, self.theme.shadow, length, thick + 2)
        P.corner_brackets(frame, x1, y1, x2, y2, color, length, thick)

    def _motion_arrow(self, frame, t: Track, box, speed, color, s) -> None:
        """
        Velocity arrow anchored on the torso (so it never runs off the frame edge):
        points right/left for sideways motion, DOWN when approaching the camera,
        UP when moving away. Longer arrow = faster.
        """
        if speed < self.theme.still_kmh or t.lateral_mps is None or t.radial_mps is None:
            return
        dx, dy = t.lateral_mps, -t.radial_mps
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            return
        ux, uy = dx / norm, dy / norm

        x1, y1, x2, y2 = box
        box_h = y2 - y1
        anchor = ((x1 + x2) / 2, y1 + 0.5 * box_h)
        length = float(np.clip(box_h * 0.12 + min(speed, 20.0) * 8.0 * s, 56 * s, max(56 * s, box_h * 0.45)))
        start = (int(anchor[0] - ux * length * 0.5), int(anchor[1] - uy * length * 0.5))
        end = (int(anchor[0] + ux * length * 0.5), int(anchor[1] + uy * length * 0.5))
        P.arrow(frame, start, end, color, max(4, int(5 * s)), int(20 * s), self.theme.shadow)

    # -------------------------------------------------------------------- card
    def _card(self, frame, t: Track, state: MotionState, speed: float,
              box: tuple[int, int, int, int], s: float) -> None:
        th = self.theme
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box

        name = t.name if t.name else f"ID {t.track_id}"
        if t.name and self.show_id_with_name:
            name += f"  #{t.track_id}"
        name_color = th.text if t.name else th.muted

        n_sc, n_th = 0.72 * s, max(1, int(round(1.4 * s)))     # name
        b_sc, b_th = 0.95 * s, max(1, int(round(1.6 * s)))     # big numbers
        u_sc, u_th = 0.46 * s, 1                               # units / captions

        speed_txt = f"{speed:.1f}"
        dist_txt = f"{t.distance_m:.1f}" if t.distance_m is not None else "--"

        trend = None
        if t.radial_mps is not None and abs(t.radial_mps) >= th.trend_min_mps:
            trend = ("closer", th.closer_color, True) if t.radial_mps < 0 else ("farther", th.farther_color, False)

        # ---- measure
        pad, bar, gap = int(11 * s), int(5 * s), int(8 * s)
        name_w, name_h = P.text_size(name, n_sc, n_th)
        state_w, _ = P.text_size(state.label, u_sc, u_th)
        sp_w, sp_h = P.text_size(speed_txt, b_sc, b_th)
        spu_w, _ = P.text_size("km/h", u_sc, u_th)
        di_w, _ = P.text_size(dist_txt, b_sc, b_th)
        diu_w, _ = P.text_size("m", u_sc, u_th)

        row2_w = sp_w + 4 + spu_w + int(18 * s) + di_w + 4 + diu_w
        if trend:
            tr_w, _ = P.text_size(trend[0], u_sc, u_th)
            row2_w += int(8 * s) + int(7 * s) * 2 + 4 + tr_w
        row1_w = name_w + int(16 * s) + state_w

        inner_w = max(row1_w, row2_w)
        card_w = bar + pad * 2 + inner_w
        card_h = pad + name_h + int(10 * s) + sp_h + pad

        cx = min(max(x1, 8), max(8, W - card_w - 8))
        cy = y1 - card_h - int(8 * s)                          # preferred: above the head
        if cy < 8:
            below = y2 + int(30 * s)                           # next best: under the feet / ground ring
            cy = below if below + card_h < H - 8 else y1 + int(8 * s)   # last resort: inside the box

        # ---- draw
        P.blend_rounded_rect(frame, cx, cy, card_w, card_h, th.card_bg, th.card_alpha, int(9 * s))
        cv2.rectangle(frame, (cx + int(2 * s), cy + int(7 * s)),
                        (cx + int(2 * s) + bar - 1, cy + card_h - int(7 * s)), state.color, -1)

        x0 = cx + bar + pad
        y_r1 = cy + pad + name_h
        P.put_text(frame, name, (x0, y_r1), n_sc, name_color, n_th)
        P.put_text(frame, state.label, (x0 + inner_w - state_w, y_r1), u_sc, state.color, u_th)

        y_r2 = y_r1 + int(10 * s) + sp_h
        x = x0
        P.put_text(frame, speed_txt, (x, y_r2), b_sc, th.text, b_th)
        x += sp_w + 4
        P.put_text(frame, "km/h", (x, y_r2), u_sc, th.muted, u_th)
        x += spu_w + int(18 * s)
        P.put_text(frame, dist_txt, (x, y_r2), b_sc, th.text, b_th)
        x += di_w + 4
        P.put_text(frame, "m", (x, y_r2), u_sc, th.muted, u_th)
        if trend:
            x += diu_w + int(8 * s) + int(7 * s)
            P.triangle(frame, x, y_r2 - sp_h // 2, int(7 * s), trend[2], trend[1])
            P.put_text(frame, trend[0], (x + int(7 * s) + 4, y_r2 - sp_h // 2 + int(5 * s)), u_sc, trend[1], u_th)

    # --------------------------------------------------------------------- HUD
    def _draw_hud(self, frame, n_tracked: int, s: float) -> None:
        th = self.theme
        sc, pad = 0.5 * s, int(10 * s)
        parts = ["LIVE", f"{self._fps:.0f} FPS", f"{n_tracked} TRACKED"]
        dot = int(5 * s)
        sizes = [P.text_size(p, sc, 1) for p in parts]
        h = sizes[0][1] + pad * 2
        w = dot * 2 + int(8 * s) + sum(sz[0] for sz in sizes) + int(22 * s) * 2 + pad * 2

        x, y = int(16 * s), int(16 * s)
        P.blend_rounded_rect(frame, x, y, w, h, th.card_bg, th.card_alpha, int(h / 2))
        cy = y + h // 2
        cv2.circle(frame, (x + pad + dot, cy), dot, th.live_color, -1, P.AA)
        tx = x + pad + dot * 2 + int(8 * s)
        for i, (part, (pw, ph)) in enumerate(zip(parts, sizes)):
            P.put_text(frame, part, (tx, cy + ph // 2), sc, th.text if i == 0 else th.muted, 1)
            tx += pw + int(22 * s)

    # ----------------------------------------------------------------- helpers
    def _tick_fps(self, now: float) -> None:
        if self._last_ts is not None and now > self._last_ts:
            inst = 1.0 / (now - self._last_ts)
            self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_ts = now

    def _prune_trails(self, now: float, active_ids: set[int]) -> None:
        for tid in [i for i in self._trails if i not in active_ids]:
            trail = self._trails[tid]
            if not trail or now - trail[-1][0] > 2.0:
                del self._trails[tid]


_default_renderer = OverlayRenderer()


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Drop-in replacement for the old function; the pipeline needs no changes."""
    return _default_renderer.draw(frame, tracks)
