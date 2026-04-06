"""
SORPanel — UI panel for Statistical Outlier Removal in LichtFeld Studio.

Embeds as a collapsible section inside the Rendering panel so it sits
naturally alongside other point-cloud tools. All state lives on the
panel instance so parameters persist across draws for the session.
"""

from __future__ import annotations

import lichtfeld as lf
from lfs_plugins.ui.state import AppState


# Reasonable parameter bounds (match PointNuker defaults)
_NB_MIN, _NB_MAX = 1, 500
_STD_MIN, _STD_MAX = 0.01, 10.0
_NB_DEFAULT = 20
_STD_DEFAULT = 1.5


class SORPanel(lf.ui.Panel):
    """
    Statistical Outlier Removal panel.

    Embedded as a collapsible section in the Rendering tab so it
    feels native to LichtFeld Studio rather than bolted on.
    """

    id = "pointnuker_sor.panel"
    label = "Statistical Outlier Removal"
    parent = "lfs.rendering"      # collapsible section inside Rendering tab
    order = 300
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    poll_dependencies = {lf.ui.PollDependency.SCENE}

    # --- Parameter state ---
    def __init__(self):
        self._nb_neighbors: int = _NB_DEFAULT
        self._std_ratio: float = _STD_DEFAULT
        self._last_result: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_scene.value

    # ------------------------------------------------------------------
    def draw(self, ui) -> None:
        # ---- Header / description ----
        ui.text_disabled("Port of PointNuker v1.0 SOR (MIT) — GS-safe")
        ui.separator()

        # ---- nb_neighbors ----
        with ui.split(0.45) as row:
            row.label("Neighbours")
            self._nb_neighbors = int(
                ui.int_slider(
                    "##nb",
                    self._nb_neighbors,
                    _NB_MIN,
                    _NB_MAX,
                )
            )
        ui.text_disabled(
            "  Neighbours used to compute each point's mean distance.\n"
            "  Higher = more context, slower."
        )

        ui.separator()

        # ---- std_ratio ----
        with ui.split(0.45) as row:
            row.label("Std Ratio")
            self._std_ratio = ui.float_slider(
                "##std",
                self._std_ratio,
                _STD_MIN,
                _STD_MAX,
            )
        ui.text_disabled(
            "  Points beyond std_ratio × σ of mean distance are removed.\n"
            "  Lower = more aggressive. Try 1.0–2.0 for most scenes."
        )

        ui.separator()

        # ---- Preset buttons ----
        ui.label("Presets")
        with ui.row() as row:
            if row.button("Conservative"):
                self._nb_neighbors = 30
                self._std_ratio = 2.0
                self._last_result = "Preset: Conservative (nb=30, std=2.0)"

            if row.button("Balanced"):
                self._nb_neighbors = 20
                self._std_ratio = 1.5
                self._last_result = "Preset: Balanced (nb=20, std=1.5)"

            if row.button("Aggressive"):
                self._nb_neighbors = 10
                self._std_ratio = 1.0
                self._last_result = "Preset: Aggressive (nb=10, std=1.0)"

        ui.separator()

        # ---- Run button ----
        if ui.button_styled("Run SOR on Scene", "primary"):
            self._run_sor()

        # ---- Restore button ----
        if ui.button("Restore Deleted Gaussians"):
            self._restore_all()

        # ---- Status line ----
        if self._last_result:
            ui.separator()
            ui.label(self._last_result)

    # ------------------------------------------------------------------
    def _run_sor(self) -> None:
        """Invoke SOROperator with current panel parameters."""
        op = lf.ops.call(
            "pointnuker_sor.sor_operator",
            nb_neighbors=self._nb_neighbors,
            std_ratio=self._std_ratio,
        )
        if op and op.get("status") == "FINISHED":
            self._last_result = (
                f"✓ SOR complete  (nb={self._nb_neighbors}, "
                f"std={self._std_ratio:.2f})"
            )
        else:
            self._last_result = "⚠ SOR skipped or failed — check the log."

    # ------------------------------------------------------------------
    def _restore_all(self) -> None:
        """Clear all soft-deleted gaussians and redraw."""
        scene = lf.get_scene()
        if scene is None:
            self._last_result = "⚠ No scene loaded."
            return

        restored = 0
        for node in scene.get_nodes():
            sd = node.splat_data()
            if sd is not None:
                sd.clear_deleted()
                restored += 1

        if restored:
            scene.notify_changed()
            self._last_result = f"✓ Restored deleted gaussians on {restored} node(s)."
        else:
            self._last_result = "No splat nodes found."
