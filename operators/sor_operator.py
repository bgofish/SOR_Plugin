"""
SOROperator — Statistical Outlier Removal operator for LichtFeld Studio.

Core algorithm ported from PointNuker v1.0 (MIT License).
Uses Open3D's remove_statistical_outlier() and maps surviving indices
back to the original splat data for GS-safe soft deletion.
"""

from __future__ import annotations

import numpy as np
import lichtfeld as lf
from lfs_plugins.types import Operator
from lfs_plugins.props import IntProperty, FloatProperty

try:
    import open3d as o3d
except ImportError as e:
    raise ImportError(
        "[pointnuker_sor] open3d is required.\n"
        "Install it with:  pip install open3d\n"
        f"Detail: {e}"
    )

# Shared result state readable by the panel
last_sor_result: dict = {}


class SOROperator(Operator):
    """
    Run Statistical Outlier Removal on the active 3DGS scene.

    Each gaussian's XYZ mean is treated as a point. Open3D identifies
    statistical outliers; those points are soft-deleted in the scene
    so all other splat attributes (SH, opacity, scale, rotation) are
    preserved intact — exactly like PointNuker's GS-safe mode.
    """

    label = "Run Statistical Outlier Removal"
    description = (
        "Remove statistical outlier gaussians using Open3D SOR. "
        "Soft-deletes outliers so all splat attributes are preserved."
    )
    options = {"UNDO"}

    # --- Parameters (match PointNuker defaults) ---
    nb_neighbors: int = IntProperty(
        default=20,
        min=1,
        max=500,
        name="Neighbours",
        description=(
            "Number of nearest neighbours to analyse per point. "
            "Higher = stricter neighbourhood context."
        ),
    )
    std_ratio: float = FloatProperty(
        default=1.5,
        min=0.01,
        max=10.0,
        name="Std Ratio",
        description=(
            "Points whose mean distance to neighbours exceeds "
            "std_ratio × global standard deviation are removed. "
            "Lower = more aggressive removal."
        ),
    )

    # ------------------------------------------------------------------
    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    # ------------------------------------------------------------------
    def execute(self, context) -> set:
        scene = lf.get_scene()
        if scene is None:
            lf.log.warn("[SOR] No scene loaded.")
            return {"CANCELLED"}

        # Collect all splat nodes
        splat_nodes = [
            node for node in scene.get_nodes()
            if node.splat_data() is not None
        ]
        if not splat_nodes:
            lf.log.warn("[SOR] No splat nodes found in scene.")
            return {"CANCELLED"}

        total_removed = 0
        total_initial = 0

        for node in splat_nodes:
            sd = node.splat_data()
            n_pts = sd.num_points
            total_initial += n_pts

            lf.log.info(
                f"[SOR] Processing '{node.name}': {n_pts:,} gaussians  "
                f"(nb_neighbors={self.nb_neighbors}, std_ratio={self.std_ratio})"
            )

            # --- Build Open3D point cloud from gaussian means ---
            means_np = sd.means_raw.cpu().numpy()  # [N, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(means_np.astype(np.float64))

            # --- Run SOR (ported directly from PointNuker._clean_proc) ---
            pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=int(self.nb_neighbors),
                std_ratio=float(self.std_ratio),
            )

            if len(inlier_indices) == 0:
                lf.log.warn(
                    f"[SOR] '{node.name}': SOR would remove ALL points — "
                    "skipping. Try raising std_ratio or lowering nb_neighbors."
                )
                continue

            # --- Build outlier mask ---
            inlier_set = set(inlier_indices)
            outlier_mask_np = np.array(
                [i not in inlier_set for i in range(n_pts)],
                dtype=bool,
            )

            removed = int(outlier_mask_np.sum())
            pct = removed / max(n_pts, 1) * 100.0
            lf.log.info(
                f"[SOR] '{node.name}': removed {removed:,} / {n_pts:,} "
                f"gaussians ({pct:.2f}%)"
            )

            # --- Soft-delete outliers (GS-safe; preserves all attributes) ---
            outlier_tensor = lf.Tensor.from_numpy(outlier_mask_np).cuda()
            sd.soft_delete(outlier_tensor)
            total_removed += removed

        scene.notify_changed()

        pct_total = total_removed / max(total_initial, 1) * 100.0
        lf.log.info(
            f"[SOR] Done — initial: {total_initial:,} | "
            f"removed: {total_removed:,} ({pct_total:.2f}%) | "
            f"remaining: {total_initial - total_removed:,}"
        )
        import pointnuker_sor.operators.sor_operator as _self_mod
        _self_mod.last_sor_result = {
            "initial": total_initial,
            "removed": total_removed,
            "remaining": total_initial - total_removed,
            "pct": pct_total,
        }
        return {"FINISHED"}
