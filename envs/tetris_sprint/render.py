"""Rendering hooks for tetris_sprint — consumed by simulacrum.viz.

Renderers contain ZERO game logic: they map a schema-conformant state JSON to
text or a matplotlib figure. They never import reference.py or fast.py.
"""

from __future__ import annotations


def render_state_text(state: dict) -> str:
    raise NotImplementedError("TODO: text rendering of one state dict")


def render_state_mpl(state: dict, ax) -> None:
    raise NotImplementedError("TODO: draw one state dict onto a matplotlib Axes")
