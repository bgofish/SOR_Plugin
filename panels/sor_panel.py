"""
SORPanel — UI panel for Statistical Outlier Removal on loaded 3DGS scenes.

Embeds as a collapsible section inside the Rendering panel. Init point cloud
pre-processing (SOR + crop on points3D files) lives in COLMAPPanel.
"""

from __future__ import annotations

import lichtfeld as lf
from lfs_plugins.ui.state import AppState

_NB_MIN, _NB_MAX   = 1, 500
_STD_MIN, _STD_MAX = 0.01, 10.0
_NB_DEFAULT        = 20
_STD_DEFAULT       = 1.5


class SORPanel(lf.ui.Panel):
    """
    Statistical Outlier Removal on Gaussian Splatting scene nodes.
    """

    id               = "pointnuker_sor.panel"
    label            = "Statistical Outlier Removal"
    parent           = "lfs.rendering"
    order            = 300
    options          = {lf.ui.PanelOption.DEFAULT_CLOSED}
    poll_dependencies = {lf.ui.PollDependency.SCENE}

    def __init__(self):
        self._nb_neighbors: int  = _NB_DEFAULT
        self._std_ratio: float   = _STD_DEFAULT
        self._last_result: str   = ""
        self._merge_name: str    = "merged"
        self._visible_only: bool = True
        self._folder_name: str   = "Group"
        self._move_target: str   = "Selection"

    # ------------------------------------------------------------------
    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_scene.value

    # ------------------------------------------------------------------
    def draw(self, ui) -> None:
        ui.text_disabled("Port of PointNuker v1.0 SOR (MIT) — V004")
        ui.separator()

        # Neighbours
        ui.label("Neighbours")
        _, self._nb_neighbors = ui.slider_int(
            "##nb", self._nb_neighbors, _NB_MIN, _NB_MAX
        )
        ui.text_disabled(
            "  Neighbours used to compute each point's mean distance.\n"
            "  Higher = more context, slower."
        )
        ui.separator()

        # Std Ratio
        ui.label("Std Ratio")
        _, self._std_ratio = ui.slider_float(
            "##std", self._std_ratio, _STD_MIN, _STD_MAX
        )
        ui.text_disabled(
            "  Points beyond std_ratio x sigma of mean distance are removed.\n"
            "  Lower = more aggressive. Try 1.0-2.0 for most scenes."
        )
        ui.separator()

        # Presets
        ui.label("Presets")
        ui.same_line()
        if ui.button("Conservative"):
            self._nb_neighbors, self._std_ratio = 30, 2.0
            self._last_result = "Preset: Conservative (nb=30, std=2.0)"
        ui.same_line()
        if ui.button("Balanced"):
            self._nb_neighbors, self._std_ratio = 20, 1.5
            self._last_result = "Preset: Balanced (nb=20, std=1.5)"
        ui.same_line()
        if ui.button("Aggressive"):
            self._nb_neighbors, self._std_ratio = 10, 1.0
            self._last_result = "Preset: Aggressive (nb=10, std=1.0)"
        ui.separator()

        # Visible-only toggle
        _, self._visible_only = ui.checkbox("Visible nodes only", self._visible_only)
        ui.separator()

        # Run
        if ui.button_styled("Run SOR on Scene", "primary"):
            self._run_sor()
        ui.separator()

        # Merge
        ui.push_item_width(120)
        _, self._merge_name = ui.input_text("##merge_name", self._merge_name)
        ui.pop_item_width()
        ui.same_line()
        if ui.button_styled("Merge Visible Nodes", "primary"):
            self._merge_visible()

        # New Group Folder
        ui.separator()
        ui.label("New Group Folder")
        ui.push_item_width(120)
        _, self._folder_name = ui.input_text("##folder_name", self._folder_name)
        ui.pop_item_width()
        ui.same_line()
        if ui.button("Create"):
            self._create_folder()
        ui.text_disabled("  Adds an empty folder node to the scene hierarchy.")
        ui.separator()

        # Move Selected Splats
        ui.label("Move Selected Splats")
        ui.push_item_width(120)
        _, self._move_target = ui.input_text("##move_target", self._move_target)
        ui.pop_item_width()
        ui.same_line()
        if ui.button("Move"):
            self._move_selected()
        ui.text_disabled("  Target node name to move selected splats into.")

        # Status
        if self._last_result:
            ui.separator()
            ui.label(self._last_result)

    # ==================================================================
    # SOR on scene gaussians
    # ==================================================================

    def _run_sor(self) -> None:
        import numpy as np
        import open3d as o3d

        self._last_result = "Running SOR on scene…"

        scene = lf.get_scene()
        if scene is None:
            self._last_result = "No scene loaded."
            return

        candidates  = scene.get_visible_nodes() if self._visible_only else scene.get_nodes()
        splat_nodes = [n for n in candidates if n.splat_data() is not None]

        if not splat_nodes:
            scope = "visible" if self._visible_only else "scene"
            self._last_result = f"No splat nodes found ({scope})."
            return

        total_removed = 0
        total_initial = 0

        for node in splat_nodes:
            sd    = node.splat_data()
            n_pts = sd.num_points
            total_initial += n_pts

            means_np = sd.means_raw.cpu().numpy()
            pcd      = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(means_np.astype(np.float64))

            _, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=int(self._nb_neighbors),
                std_ratio=float(self._std_ratio),
            )

            if len(inlier_indices) == 0:
                self._last_result = f"'{node.name}': SOR would remove ALL points — skipped."
                continue

            inlier_set      = set(inlier_indices)
            outlier_mask_np = np.array([i not in inlier_set for i in range(n_pts)], dtype=bool)
            outlier_idx     = np.where(outlier_mask_np)[0]
            removed         = int(outlier_mask_np.sum())
            total_removed  += removed

            def gather_rows(tensor, idx=outlier_idx):
                return lf.Tensor.from_numpy(tensor.cpu().numpy()[idx]).cuda()

            scene.add_splat(
                f"{node.name}_outliers",
                gather_rows(sd.means_raw), gather_rows(sd.sh0_raw),
                gather_rows(sd.shN_raw),   gather_rows(sd.scaling_raw),
                gather_rows(sd.rotation_raw), gather_rows(sd.opacity_raw),
                sd.active_sh_degree, sd.scene_scale,
            )

            inlier_idx = np.where(~outlier_mask_np)[0]
            def gather_inliers(tensor, idx=inlier_idx):
                return lf.Tensor.from_numpy(tensor.cpu().numpy()[idx]).cuda()

            original_name   = node.name
            inlier_means    = gather_inliers(sd.means_raw)
            inlier_sh0      = gather_inliers(sd.sh0_raw)
            inlier_shN      = gather_inliers(sd.shN_raw)
            inlier_scaling  = gather_inliers(sd.scaling_raw)
            inlier_rotation = gather_inliers(sd.rotation_raw)
            inlier_opacity  = gather_inliers(sd.opacity_raw)
            active_sh       = sd.active_sh_degree
            s_scale         = sd.scene_scale

            scene.remove_node(original_name)
            scene.add_splat(
                original_name,
                inlier_means, inlier_sh0, inlier_shN,
                inlier_scaling, inlier_rotation, inlier_opacity,
                active_sh, s_scale,
            )

        scene.invalidate_cache()
        scene.notify_changed()

        pct       = total_removed / max(total_initial, 1) * 100.0
        remaining = total_initial - total_removed
        scope     = "visible" if self._visible_only else "all"
        self._last_result = (
            f"Removed {total_removed:,} / {total_initial:,} gaussians "
            f"({pct:.1f}%) — {remaining:,} remaining [{scope}]"
        )

    # ==================================================================
    # Merge visible nodes
    # ==================================================================

    def _merge_visible(self) -> None:
        scene = lf.get_scene()
        if scene is None:
            self._last_result = "No scene loaded."
            return

        nodes = [n for n in scene.get_visible_nodes() if n.splat_data() is not None]
        if not nodes:
            self._last_result = "No visible splat nodes to merge."
            return

        name     = self._merge_name.strip() or "merged"
        group_id = scene.add_group(name)
        for n in nodes:
            scene.reparent(n.id, group_id)

        scene.merge_group(name)
        scene.notify_changed()
        self._last_result = f"Merged {len(nodes)} node(s) into '{name}'."

    # ==================================================================
    # Create group folder
    # ==================================================================

    def _create_folder(self) -> None:
        name = self._folder_name.strip() or "Group"
        scene = lf.get_scene()
        if scene is None:
            self._last_result = "No scene loaded."
            return
        try:
            scene.add_group(name)
            scene.notify_changed()
            self._last_result = f"Created group '{name}'."
        except Exception as e:
            self._last_result = f"Create group failed: {e}"

    # ==================================================================
    # Move selected splats
    # ==================================================================

    def _move_selected(self) -> None:
        import numpy as np

        target_name = self._move_target.strip()
        if not target_name:
            self._last_result = "Enter a target node name first."
            return

        scene = lf.get_scene()
        if scene is None:
            self._last_result = "No scene loaded."
            return

        global_mask = scene.selection_mask
        if global_mask is None or not lf.has_selection():
            self._last_result = "Nothing selected."
            return

        visible_splat_nodes = [n for n in scene.get_visible_nodes()
                               if n.splat_data() is not None]

        # Find the source node that owns the current selection
        start_idx  = 0
        source_node = None
        for n in visible_splat_nodes:
            nd = n.splat_data().num_points
            local_mask = global_mask[start_idx : start_idx + nd].cpu().numpy().astype(bool)
            if local_mask.sum() > 0:
                source_node = n
                break
            start_idx += nd

        if source_node is None:
            self._last_result = "No splats selected in any visible node."
            return

        source_name = source_node.name
        num_points  = source_node.splat_data().num_points
        local_mask  = global_mask[start_idx : start_idx + num_points].cpu().numpy().astype(bool)
        selected_count = int(local_mask.sum())

        scene.clear_selection()

        source_sd    = source_node.splat_data()
        selected_idx = np.where(local_mask)[0]
        inlier_idx   = np.where(~local_mask)[0]
        active_sh    = source_sd.active_sh_degree
        s_scale      = source_sd.scene_scale

        def _gather(tensor, idx):
            return lf.Tensor.from_numpy(tensor.cpu().numpy()[idx]).cuda()

        sel_means    = _gather(source_sd.means_raw,    selected_idx)
        sel_sh0      = _gather(source_sd.sh0_raw,      selected_idx)
        sel_shN      = _gather(source_sd.shN_raw,      selected_idx)
        sel_scaling  = _gather(source_sd.scaling_raw,  selected_idx)
        sel_rotation = _gather(source_sd.rotation_raw, selected_idx)
        sel_opacity  = _gather(source_sd.opacity_raw,  selected_idx)

        inlier_means    = _gather(source_sd.means_raw,    inlier_idx)
        inlier_sh0      = _gather(source_sd.sh0_raw,      inlier_idx)
        inlier_shN      = _gather(source_sd.shN_raw,      inlier_idx)
        inlier_scaling  = _gather(source_sd.scaling_raw,  inlier_idx)
        inlier_rotation = _gather(source_sd.rotation_raw, inlier_idx)
        inlier_opacity  = _gather(source_sd.opacity_raw,  inlier_idx)

        target_node = scene.get_node(target_name)
        if target_node is not None:
            target_sd = target_node.splat_data()
            def _cat(a, b):
                return lf.Tensor.from_numpy(
                    np.concatenate([a.cpu().numpy(), b.cpu().numpy()], axis=0)
                ).cuda()
            scene.remove_node(target_name)
            scene.add_splat(
                target_name,
                _cat(target_sd.means_raw,    sel_means),
                _cat(target_sd.sh0_raw,      sel_sh0),
                _cat(target_sd.shN_raw,      sel_shN),
                _cat(target_sd.scaling_raw,  sel_scaling),
                _cat(target_sd.rotation_raw, sel_rotation),
                _cat(target_sd.opacity_raw,  sel_opacity),
                active_sh, s_scale,
            )
        else:
            scene.add_splat(
                target_name,
                sel_means, sel_sh0, sel_shN,
                sel_scaling, sel_rotation, sel_opacity,
                active_sh, s_scale,
            )

        scene.remove_node(source_name)
        scene.add_splat(
            source_name,
            inlier_means, inlier_sh0, inlier_shN,
            inlier_scaling, inlier_rotation, inlier_opacity,
            active_sh, s_scale,
        )

        scene.invalidate_cache()
        scene.notify_changed()
        self._last_result = (
            f"Moved {selected_count:,} splats from '{source_name}' → '{target_name}'."
        )
        self._move_target = "Selection"

    # ==================================================================
    # Restore soft-deleted gaussians
    # ==================================================================

    def _restore_all(self) -> None:
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
