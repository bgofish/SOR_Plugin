"""
COLMAPPanel — Pre-processing tools for COLMAP init point clouds.

Sits inside the Training tab (parent = lfs.training).
Covers:
  - Statistical Outlier Removal on points3D.bin / points3D.txt
  - Axis-aligned bounding-box crop
  - Sphere crop
  - Crop + SOR in one click
"""

from __future__ import annotations

import lichtfeld as lf
from lfs_plugins.ui.state import AppState

_NB_MIN, _NB_MAX   = 1, 500
_STD_MIN, _STD_MAX = 0.01, 10.0
_NB_DEFAULT        = 20
_STD_DEFAULT       = 1.5

_CROP_AABB   = "AABB"
_CROP_SPHERE = "Sphere"


class COLMAPPanel(lf.ui.Panel):
    """
    COLMAP Tools — pre-processing for Gaussian Splatting init point clouds.
    """

    id                = "pointnuker_sor.colmap_panel"
    label             = "COLMAP Tools"
    parent            = "lfs.training"
    order             = 50
    options           = {lf.ui.PanelOption.DEFAULT_CLOSED}
    poll_dependencies = {lf.ui.PollDependency.SCENE}

    # ------------------------------------------------------------------
    def __init__(self):
        # File path
        self._colmap_path: str    = ""
        self._path_input_buf: str = ""   # separate ImGui buffer for input_text
        self._file_info: str      = ""   # e.g. "12,450 points  X[-2.1, 3.4] ..."

        # SOR params
        self._nb_neighbors: int   = _NB_DEFAULT
        self._std_ratio: float    = _STD_DEFAULT

        # Crop params
        self._crop_mode: str      = _CROP_AABB
        self._aabb_min_x: float   = -10.0
        self._aabb_max_x: float   =  10.0
        self._aabb_min_y: float   = -10.0
        self._aabb_max_y: float   =  10.0
        self._aabb_min_z: float   = -10.0
        self._aabb_max_z: float   =  10.0
        self._sphere_cx: float    =   0.0
        self._sphere_cy: float    =   0.0
        self._sphere_cz: float    =   0.0
        self._sphere_r: float     =  10.0

        # Pending path set by folder browser (applied before input_text renders)
        self._pending_path: str   = ""

        # Status
        self._last_result: str    = ""

    # ------------------------------------------------------------------
    @classmethod
    def poll(cls, context) -> bool:
        return True   # no scene required for file ops

    # ------------------------------------------------------------------
    def draw(self, ui) -> None:

        ui.text_disabled("Pre-process COLMAP sparse point clouds before training.")
        ui.separator()

        # ================================================================
        # File picker
        # ================================================================
        ui.label("points3D file")

        # Browse button — opens folder dialog, auto-detects file
        if ui.button("Browse...##browse"):
            folder = lf.ui.open_folder_dialog(
                title="Select sparse/0 folder",
                start_dir=self._colmap_path or "",
            )
            if folder:
                import os
                for name in ("points3D.bin", "points3D.txt"):
                    candidate = os.path.join(folder, name)
                    if os.path.isfile(candidate):
                        self._colmap_path    = candidate
                        self._path_input_buf = candidate
                        self._file_info      = ""
                        self._last_result    = f"File: {candidate}"
                        break
                else:
                    self._last_result = (
                        "No points3D.bin or points3D.txt found in that folder."
                    )

        # Editable path — manual override (same line as Browse button)
        ui.same_line()
        ui.push_item_width(-1)
        changed, new_val = ui.input_text("##colmap_path", self._path_input_buf)
        ui.pop_item_width()
        if changed:
            self._path_input_buf = new_val
            self._colmap_path    = new_val
            self._file_info      = ""

        # Inspect button — reads file, shows count + actual extents
        if ui.button("Inspect File"):
            self._inspect_file()
        if self._file_info:
            ui.text_disabled(f"  {self._file_info}")

        ui.separator()
        ui.label("Statistical Outlier Removal")
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




        # ================================================================
        # SOR parameters
        # ================================================================


        ui.label("Neighbours")
        ui.push_item_width(-1)
        _, self._nb_neighbors = ui.slider_int(
            "##nb", self._nb_neighbors, _NB_MIN, _NB_MAX
        )
        ui.pop_item_width()
        ui.text_disabled(
            "  Number of neighbours used to compute mean distance.\n"
            "  Higher = more context, slower."
        )

        ui.label("Std Ratio")
        ui.push_item_width(-1)
        _, self._std_ratio = ui.slider_float(
            "##std", self._std_ratio, _STD_MIN, _STD_MAX
        )
        ui.pop_item_width()
        ui.text_disabled(
            "  Points beyond std_ratio x sigma are removed.\n"
            "  Lower = more aggressive. Try 1.0-2.0 for most scenes."
        )
        ui.separator()       
        if ui.button_styled("Run SOR", "secondary"):
            self._run_sor()
        ui.separator()
        ui.separator()
        # ================================================================
        # Crop parameters
        # ================================================================
        ui.label("Crop Initial Points by Axis Aligned Bounding Box [AABB] or Sphere:")
        # ui.separator()
        ui.label(" ")
        # Mode toggle
        ui.label("Mode")
        ui.same_line()
        if ui.button(f"[{'x' if self._crop_mode == _CROP_AABB else ' '}] AABB"):
            self._crop_mode = _CROP_AABB
        ui.same_line()
        if ui.button(f"[{'x' if self._crop_mode == _CROP_SPHERE else ' '}] Sphere"):
            self._crop_mode = _CROP_SPHERE

        # Fit bounds to actual data
        ui.same_line()
        if ui.button("Fit to File"):
            self._fit_bounds()
        ui.separator()

        if self._crop_mode == _CROP_AABB:
            ui.label("X")
            ui.push_item_width(-1)
            _, self._aabb_min_x = ui.input_float("##xmin", self._aabb_min_x, 0.0, 0.0, "min  %.3f")
            _, self._aabb_max_x = ui.input_float("##xmax", self._aabb_max_x, 0.0, 0.0, "max  %.3f")
            ui.pop_item_width()
            ui.label("Y")
            ui.push_item_width(-1)
            _, self._aabb_min_y = ui.input_float("##ymin", self._aabb_min_y, 0.0, 0.0, "min  %.3f")
            _, self._aabb_max_y = ui.input_float("##ymax", self._aabb_max_y, 0.0, 0.0, "max  %.3f")
            ui.pop_item_width()
            ui.label("Z")
            ui.push_item_width(-1)
            _, self._aabb_min_z = ui.input_float("##zmin", self._aabb_min_z, 0.0, 0.0, "min  %.3f")
            _, self._aabb_max_z = ui.input_float("##zmax", self._aabb_max_z, 0.0, 0.0, "max  %.3f")
            ui.pop_item_width()
        else:
            ui.label("Centre")
            ui.push_item_width(-1)
            _, self._sphere_cx = ui.input_float("##cx", self._sphere_cx, 0.0, 0.0, "X  %.3f")
            _, self._sphere_cy = ui.input_float("##cy", self._sphere_cy, 0.0, 0.0, "Y  %.3f")
            _, self._sphere_cz = ui.input_float("##cz", self._sphere_cz, 0.0, 0.0, "Z  %.3f")
            ui.pop_item_width()
            ui.label("Radius")
            ui.push_item_width(-1)
            _, self._sphere_r = ui.input_float("##sr", self._sphere_r, 0.0, 0.0, "%.3f")
            ui.pop_item_width()

        ui.separator()
        if ui.button_styled("Crop", "secondary"):
            self._crop(then_sor=False)
        ui.same_line()
        if ui.button_styled("Crop + SOR", "primary"):
            self._crop(then_sor=True)

        # ================================================================
        # Status
        # ================================================================
        if self._last_result:
            ui.separator()
            ui.label(self._last_result)

    # ==================================================================
    # Shared file I/O
    # ==================================================================

    @staticmethod
    def _read_points(path: str) -> dict:
        """Read points3D.bin or .txt → {pid: (xyz, rgb, error, track)}."""
        import struct
        ext = path.rsplit(".", 1)[-1].lower()
        pts: dict = {}
        if ext == "bin":
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                for _ in range(n):
                    (pid,)  = struct.unpack("<Q", f.read(8))
                    xyz     = struct.unpack("<3d", f.read(24))
                    rgb     = struct.unpack("<3B", f.read(3))
                    (err,)  = struct.unpack("<d", f.read(8))
                    (tlen,) = struct.unpack("<Q", f.read(8))
                    track   = struct.unpack(f"<{2*tlen}I", f.read(8 * tlen))
                    pts[pid] = (xyz, rgb, err, track)
        else:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tokens = line.split()
                    pid   = int(tokens[0])
                    xyz   = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
                    rgb   = (int(tokens[4]),   int(tokens[5]),   int(tokens[6]))
                    err   = float(tokens[7])
                    track = tuple(int(t) for t in tokens[8:])
                    pts[pid] = (xyz, rgb, err, track)
        return pts

    @staticmethod
    def _write_points(path: str, pts: dict, keep_pids: set) -> None:
        """Write surviving points back to path in the original format."""
        import struct
        ext     = path.rsplit(".", 1)[-1].lower()
        ordered = [pid for pid in pts if pid in keep_pids]
        if ext == "bin":
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(ordered)))
                for pid in ordered:
                    xyz, rgb, err, track = pts[pid]
                    tlen = len(track) // 2
                    f.write(struct.pack("<Q",  pid))
                    f.write(struct.pack("<3d", *xyz))
                    f.write(struct.pack("<3B", *rgb))
                    f.write(struct.pack("<d",  err))
                    f.write(struct.pack("<Q",  tlen))
                    f.write(struct.pack(f"<{len(track)}I", *track))
        else:
            with open(path, "w") as f:
                f.write("# 3D point list with one line of data per point:\n")
                f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                f.write(f"# Number of points: {len(ordered)}\n")
                for pid in ordered:
                    xyz, rgb, err, track = pts[pid]
                    track_str = " ".join(str(t) for t in track)
                    f.write(
                        f"{pid} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} "
                        f"{rgb[0]} {rgb[1]} {rgb[2]} {err:.10f}"
                        + (f" {track_str}" if track_str else "")
                        + "\n"
                    )

    def _validate_path(self) -> str | None:
        import os
        path = self._colmap_path.strip()
        if not path:
            self._last_result = "No file path set."
            return None
        if not os.path.isfile(path):
            self._last_result = f"File not found — {path}"
            return None
        if os.path.splitext(path)[1].lower() not in (".bin", ".txt"):
            self._last_result = "Expected a .bin or .txt file."
            return None
        return path

    def _backup(self, path: str) -> bool:
        import shutil
        try:
            shutil.copy2(path, path + ".bak")
            return True
        except Exception as exc:
            self._last_result = f"Backup failed — {exc}"
            return False

    def _load_xyzs(self) -> tuple[dict, object] | tuple[None, None]:
        """Load points and return (points_dict, xyzs_ndarray), or (None, None) on error."""
        import numpy as np
        path = self._validate_path()
        if path is None:
            return None, None
        try:
            points = self._read_points(path)
        except Exception as exc:
            self._last_result = f"Read error — {exc}"
            return None, None
        if not points:
            self._last_result = "File is empty."
            return None, None
        pids = list(points.keys())
        xyzs = np.array([points[p][0] for p in pids], dtype=np.float64)
        return points, xyzs

    # ==================================================================
    # Inspect file — show point count and actual extents
    # ==================================================================

    def _inspect_file(self) -> None:
        points, xyzs = self._load_xyzs()
        if points is None:
            return
        mn = xyzs.min(axis=0)
        mx = xyzs.max(axis=0)
        self._file_info = (
            f"{len(points):,} pts  "
            f"X[{mn[0]:.2f}, {mx[0]:.2f}]  "
            f"Y[{mn[1]:.2f}, {mx[1]:.2f}]  "
            f"Z[{mn[2]:.2f}, {mx[2]:.2f}]"
        )
        self._last_result = "File inspected — extents shown above."

    # ==================================================================
    # Fit bounds to actual data extents
    # ==================================================================

    def _fit_bounds(self) -> None:
        points, xyzs = self._load_xyzs()
        if points is None:
            return
        mn = xyzs.min(axis=0)
        mx = xyzs.max(axis=0)
        cx = xyzs.mean(axis=0)

        if self._crop_mode == _CROP_AABB:
            self._aabb_min_x, self._aabb_max_x = float(mn[0]), float(mx[0])
            self._aabb_min_y, self._aabb_max_y = float(mn[1]), float(mx[1])
            self._aabb_min_z, self._aabb_max_z = float(mn[2]), float(mx[2])
            self._last_result = (
                f"AABB fitted to data:  "
                f"X[{mn[0]:.3f}, {mx[0]:.3f}]  "
                f"Y[{mn[1]:.3f}, {mx[1]:.3f}]  "
                f"Z[{mn[2]:.3f}, {mx[2]:.3f}]"
            )
        else:
            import numpy as np
            self._sphere_cx, self._sphere_cy, self._sphere_cz = (
                float(cx[0]), float(cx[1]), float(cx[2])
            )
            # Radius = distance from centroid to furthest point
            dists = np.linalg.norm(xyzs - cx, axis=1)
            self._sphere_r = float(dists.max())
            self._last_result = (
                f"Sphere fitted to data:  "
                f"centre ({cx[0]:.3f}, {cx[1]:.3f}, {cx[2]:.3f})  "
                f"radius {self._sphere_r:.3f}"
            )

    # ==================================================================
    # SOR
    # ==================================================================

    def _run_sor(self, points: dict | None = None) -> dict | None:
        """Run SOR on the file (or a pre-loaded dict for Crop+SOR chaining).

        Returns surviving points dict, or None on failure.
        When called standalone (points=None) writes result back to disk.
        """
        import numpy as np
        import open3d as o3d

        standalone = points is None

        if standalone:
            path = self._validate_path()
            if path is None:
                return None
            self._last_result = "Running SOR…"
            try:
                points = self._read_points(path)
            except Exception as exc:
                self._last_result = f"Read error — {exc}"
                return None
            if not points:
                self._last_result = "File is empty."
                return None

        n_initial = len(points)
        pids      = list(points.keys())
        xyzs      = np.array([points[p][0] for p in pids], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        _, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=int(self._nb_neighbors),
            std_ratio=float(self._std_ratio),
        )

        if len(inlier_indices) == 0:
            self._last_result = (
                "SOR would remove ALL points — skipped. "
                "Try raising Std Ratio or lowering Neighbours."
            )
            return None

        inlier_pids = {pids[i] for i in inlier_indices}
        surviving   = {pid: points[pid] for pid in inlier_pids}
        removed     = n_initial - len(inlier_pids)

        if standalone:
            if not self._backup(path):
                return None
            try:
                self._write_points(path, points, inlier_pids)
            except Exception as exc:
                self._last_result = f"Write error — {exc}"
                return None
            pct = removed / max(n_initial, 1) * 100.0
            self._last_result = (
                f"SOR: removed {removed:,} / {n_initial:,} ({pct:.1f}%) — "
                f"{len(inlier_pids):,} remaining. Backup: .bak"
            )

        return surviving

    # ==================================================================
    # Crop (+ optional SOR)
    # ==================================================================

    def _crop(self, then_sor: bool = False) -> None:
        import numpy as np

        path = self._validate_path()
        if path is None:
            return

        op = f"Crop ({self._crop_mode})" + (" + SOR" if then_sor else "")
        self._last_result = f"Running {op}…"

        try:
            points = self._read_points(path)
        except Exception as exc:
            self._last_result = f"Read error — {exc}"
            return

        if not points:
            self._last_result = "File is empty."
            return

        n_initial = len(points)
        pids      = list(points.keys())
        xyzs      = np.array([points[p][0] for p in pids], dtype=np.float64)

        # Build keep mask
        if self._crop_mode == _CROP_AABB:
            mask = (
                (xyzs[:, 0] >= self._aabb_min_x) & (xyzs[:, 0] <= self._aabb_max_x) &
                (xyzs[:, 1] >= self._aabb_min_y) & (xyzs[:, 1] <= self._aabb_max_y) &
                (xyzs[:, 2] >= self._aabb_min_z) & (xyzs[:, 2] <= self._aabb_max_z)
            )
        else:
            dx   = xyzs[:, 0] - self._sphere_cx
            dy   = xyzs[:, 1] - self._sphere_cy
            dz   = xyzs[:, 2] - self._sphere_cz
            mask = (dx*dx + dy*dy + dz*dz) <= (self._sphere_r ** 2)

        # Convert numpy bool array to a plain Python list to avoid any
        # numpy scalar truthiness edge cases when building the keep set
        mask_list    = mask.tolist()
        keep_pids    = {pids[i] for i, k in enumerate(mask_list) if k}
        crop_removed = n_initial - len(keep_pids)

        if not keep_pids:
            self._last_result = (
                "Crop region contains NO points — skipped. "
                "Use 'Inspect File' to check actual extents, then 'Fit to File'."
            )
            return

        # Optionally chain SOR on the cropped subset
        sor_removed = 0
        if then_sor:
            cropped   = {pid: points[pid] for pid in keep_pids}
            surviving = self._run_sor(points=cropped)
            if surviving is None:
                return
            sor_removed = len(keep_pids) - len(surviving)
            keep_pids   = set(surviving.keys())

        if not self._backup(path):
            return
        try:
            self._write_points(path, points, keep_pids)
        except Exception as exc:
            self._last_result = f"Write error — {exc}"
            return

        total  = crop_removed + sor_removed
        remain = n_initial - total
        pct    = total / max(n_initial, 1) * 100.0

        if then_sor:
            self._last_result = (
                f"{op}: crop removed {crop_removed:,}, SOR removed {sor_removed:,} — "
                f"{total:,} / {n_initial:,} total ({pct:.1f}%) — "
                f"{remain:,} remaining. Backup: .bak"
            )
        else:
            self._last_result = (
                f"{op}: removed {crop_removed:,} / {n_initial:,} ({pct:.1f}%) — "
                f"{remain:,} remaining. Backup: .bak"
            )
