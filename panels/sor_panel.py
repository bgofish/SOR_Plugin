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
        self._ui = ui
        # ---- Header / description ----
        ui.text_disabled("Port of PointNuker v1.0 SOR (MIT) — GS-safe")
        ui.separator()

        # ---- nb_neighbors ----
        ui.label("Neighbours")
        _, self._nb_neighbors = ui.slider_int(
            "##nb", self._nb_neighbors, _NB_MIN, _NB_MAX
        )
        ui.text_disabled(
            "  Neighbours used to compute each point's mean distance.\n"
            "  Higher = more context, slower."
        )

        ui.separator()

        # ---- std_ratio ----
        ui.label("Std Ratio")
        _, self._std_ratio = ui.slider_float(
            "##std", self._std_ratio, _STD_MIN, _STD_MAX
        )
        ui.text_disabled(
            "  Points beyond std_ratio x sigma of mean distance are removed.\n"
            "  Lower = more aggressive. Try 1.0-2.0 for most scenes."
        )

        ui.separator()

        # ---- Preset buttons ----
        ui.label("Presets")
        ui.same_line()
        if ui.button("Conservative"):
            self._nb_neighbors = 30
            self._std_ratio = 2.0
            self._last_result = "Preset: Conservative (nb=30, std=2.0)"
        ui.same_line()
        if ui.button("Balanced"):
            self._nb_neighbors = 20
            self._std_ratio = 1.5
            self._last_result = "Preset: Balanced (nb=20, std=1.5)"
        ui.same_line()
        if ui.button("Aggressive"):
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
        """Run SOR, soft-delete outliers from original, and move them to a new node."""
        import numpy as np
        import open3d as o3d

        scene = lf.get_scene()
        if scene is None:
            self._last_result = "No scene loaded."
            return

        splat_nodes = [
            node for node in scene.get_nodes()
            if node.splat_data() is not None
        ]
        if not splat_nodes:
            self._last_result = "No splat nodes found in scene."
            return

        total_removed = 0
        total_initial = 0

        for node in splat_nodes:
            sd = node.splat_data()
            n_pts = sd.num_points
            total_initial += n_pts

            means_np = sd.means_raw.cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(means_np.astype(np.float64))

            pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=int(self._nb_neighbors),
                std_ratio=float(self._std_ratio),
            )

            if len(inlier_indices) == 0:
                self._last_result = (
                    f"'{node.name}': SOR would remove ALL points - skipped."
                )
                continue

            # Build masks
            inlier_set = set(inlier_indices)
            outlier_mask_np = np.array(
                [i not in inlier_set for i in range(n_pts)], dtype=bool
            )
            outlier_idx = np.where(outlier_mask_np)[0]
            removed = int(outlier_mask_np.sum())
            total_removed += removed

            # Slice each tensor to only the outlier rows and upload to cuda
            def gather_rows(tensor, idx=outlier_idx):
                arr = tensor.cpu().numpy()
                return lf.Tensor.from_numpy(arr[idx]).cuda()

            # Create new splat node containing only the outliers
            scene.add_splat(
                f"{node.name}_outliers",
                gather_rows(sd.means_raw),
                gather_rows(sd.sh0_raw),
                gather_rows(sd.shN_raw),
                gather_rows(sd.scaling_raw),
                gather_rows(sd.rotation_raw),
                gather_rows(sd.opacity_raw),
                sd.active_sh_degree,
                sd.scene_scale,
            )

            # Soft-delete outliers from the original node
            outlier_tensor = lf.Tensor.from_numpy(outlier_mask_np).cuda()
            sd.soft_delete(outlier_tensor)

        scene.notify_changed()

        pct = total_removed / max(total_initial, 1) * 100.0
        remaining = total_initial - total_removed
        self._last_result = (
            f"Removed {total_removed:,} / {total_initial:,} gaussians "
            f"({pct:.1f}%) - {remaining:,} remaining, outliers in new node"
        )

    # ------------------------------------------------------------------
    def _restore_all(self) -> None:
        """Clear all soft-deleted gaussians and redraw."""
        scene = lf.get_scene()
        if scene is None:
            self._last_result = "No scene loaded."
            return

        restored = 0
        for node in scene.get_nodes():
            sd = node.splat_data()
            if sd is not None:
                sd.clear_deleted()
                restored += 1

        if restored:
            scene.notify_changed()
            self._last_result = f"Restored deleted gaussians on {restored} node(s)."
        else:
            self._last_result = "No splat nodes found."