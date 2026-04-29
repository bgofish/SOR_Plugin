# LichtFeld Studio | COLMAP Point Editor
### `PyVista_ColmapBIN-CRS.py` — v0.1.0

A desktop tool for loading, visualising, transforming, cropping and re-exporting COLMAP sparse reconstruction data. Built with PyVista + PyQt5, designed to work directly with LichtFeld Studio output.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Interface Overview](#interface-overview)
5. [Features](#features)
   - [Import Project](#import-project)
   - [View Presets](#view-presets)
   - [Global SRT Transform](#global-srt-transform)
   - [Point Picking Tools](#point-picking-tools)
   - [Visuals](#visuals)
   - [Cropping](#cropping)
   - [Histogram Panel](#histogram-panel)
   - [Bake SRT](#bake-srt)
   - [Export COLMAP](#export-colmap)
   - [Settings](#settings)
6. [File Reference](#file-reference)
   - [settings.json](#settingsjson)
   - [logfile.json](#logfilejson)
7. [Coordinate System Notes](#coordinate-system-notes)
8. [Supported Camera Models](#supported-camera-models)
9. [Workflow Examples](#workflow-examples)
10. [Known Limitations](#known-limitations)

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.9 + |
| PyQt5 | 5.15 + |
| pyvista | 0.43 + |
| pyvistaqt | 0.11 + |
| numpy | 1.24 + |
| scipy | 1.11 + |
| vtk | 9.2 + |

---

## Installation

```bash
pip install pyqt5 pyvista pyvistaqt numpy scipy vtk
```

---

## Quick Start

```bash
python PyVista_ColmapBIN-CRS.py
```

1. Click **📂 Import Project** and select a folder containing COLMAP sparse output files (`points3D.bin`, `images.bin`, `cameras.bin` — or the `.txt` equivalents).
2. Use the **View presets** (V01–V09) to orient the scene.
3. Adjust **SRT sliders** to rotate, translate and scale.
4. Click **Floor (3Pt)** to level the scene interactively.
5. Crop unwanted regions using the **Cropping** panel and histogram.
6. Click **✅ Bake SRT** to commit the transform.
7. Select **BIN** or **TXT** output format and click **💾 Export COLMAP**.

---

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Sidebar (left)          │  3D Viewport (centre)                │
│  ─────────────────────   │  ─────────────────────────────────── │
│  System                  │                                      │
│    Read | Write | N++    │         Point Cloud Display          │
│    📂 Import Project     │                                      │
│    ✅ Bake SRT           │                                      │
│    💾 Export  [BIN|TXT]  │                                      │
│                          │                                      │
│  View & Alignment        │                                      │
│    NW  BACK NE           │                                      │
│   LEFT TOP RIGHT         │                                      │
│    SW  FRONT SE          │                                      │
│                          │                                      │
│    FOV spinner           │                                      │
│    Floor (3Pt)           │                                      │
│    Y-Align (2Pt)         │                                      │
│    XYZ Origin            │                                      │
│    XZ  Origin            │                                      │
│                          │                                      │
│  Global SRT              │  ─────────────────────────────────── │
│    Scale                 │  Histogram Panel                     │
│    RotX  RotY  RotZ      │  ─────────────────────────────────── │
│    TransX TransY TransZ  │  Cropping                            │
│    Step Size             │    Axis Sel | Auto99 | Auto95 | Reset│
│                          │    MIN ─────────────────────────     │
│  Visuals                 │    MAX ─────────────────────────     │
│    [Origin Axis]         │                                      │
│    Point Size            │                                      │
│    Color Mode            │                                      │
└─────────────────────────────────────────────────────────────────┘
```
---

## Features

### Import Project

**Button:** `📂 Import Project`

Select a folder containing any combination of COLMAP output files. The loader auto-detects binary vs text format.

**Binary files loaded:**
- `points3D.bin` — 3D point positions, RGB colours, reprojection errors, full track data
- `images.bin` — camera poses (quaternion + translation), 2D–3D correspondences
- `cameras.bin` — intrinsic parameters for each camera model

**Text files loaded (fallback):**
- `points3D.txt`

> **Note:** `images.bin` and `cameras.bin` are optional. If absent, only point cloud operations are available; camera pose export will produce empty image/camera files.

---

### View Presets

Nine preset camera positions arranged in a 3×3 grid. Each positions the viewport camera at a fixed angle relative to the scene origin:

| C1 | C2 |C3|
|---|---|---|
|7=NW  | 8=BACK |  9=NE|
|4=LEFT |5=TOP   | 6=RIGHT|
|1=SW  | 2=FRONT | 3=SE|

**FOV spinner** — adjusts the camera field of view (10°–120°, default 45°). Changes apply immediately to the viewport.

---

### Global SRT Transform

Seven sliders with linked spinboxes control the world-space transform applied to the point cloud in real time:

| Control | Range | Default | Description |
|---|---|---|---|
| Scale | 0.01 – 20.0 | 1.0 | Uniform scale |
| RotX | −180° – +180° | 0° | Pitch (tilt forward/back) |
| RotY | −180° – +180° | 0° | Yaw (rotate left/right) |
| RotZ | −180° – +180° | 0° | Roll |
| TransX | −5000 – +5000 | 0 | Translate along X |
| TransY | −5000 – +5000 | 0 | Translate along Y |
| TransZ | −5000 – +5000 | 0 | Translate along Z |

**Step Size spinner** — controls the spinbox single-step increment for fine vs coarse adjustment.

**Double-click any slider** to snap it back to its default value.

**Live preview** — the viewport updates ~50 ms after you stop moving a slider (debounced to avoid stutter on large scenes). The Bake button turns **red** whenever the SRT is non-identity, reminding you to bake before exporting.

> Rotation convention: Euler XYZ extrinsic, applied as Rz @ Ry @ Rx. Display space has Y-axis negated relative to COLMAP space (see [Coordinate System Notes](#coordinate-system-notes)).

---

### Point Picking Tools

Four interactive picking modes. Click the button to activate — the viewport enters pick mode and the button highlights. After picking the required number of points, the result is applied automatically and the mode exits.

#### Floor (3Pt) — `plane` mode

Pick **3 points** on a flat surface (e.g. floor, table, ground plane). The tool:

1. Computes the plane normal from the cross product of the two edge vectors.
2. Ensures the normal points upward (flip [F] if needed).
3. Computes the exact Rodrigues rotation to align that normal to display +Y.
4. Extracts Euler XYZ angles and adds them to RotX, RotY, RotZ.

Here's exactly how it works:

	After placing all 3 points, the arrow appears as normal (defaulting upward)
	Press F at any time before confirming — the arrow instantly flips direction and redraws in the viewport
	Press F again to flip back — it toggles, so you can go back and forth
	Left-click confirms using whichever direction the arrow is currently pointing
	Right-click cancels and cleans up as before
	The hint text in the viewport also updated to show [F] Flip normal

**Result:** the picked surface becomes horizontal.

#### Y-Align (2Pt) — `align` mode

Pick **2 points** along a straight edge (e.g. a wall, road, fence line). Computes the horizontal bearing angle and applies it as a RotY correction to align that edge to the Z axis.

#### XYZ Origin — `xyz_origin` mode

Pick **1 point**. Subtracts its X, Y, Z coordinates from TransX, TransY, TransZ so that point becomes the world origin.

#### XZ Origin — `xz_origin` mode

Pick **1 point**. Moves only TransX and TransZ (leaves TransY unchanged), centering the scene horizontally without affecting height.

---

### Visuals

**[Origin Axis]** — toggleable button. When on, draws a colour-coded axis gizmo at the world origin scaled to 30% of the scene width:
- Red → +X
- Green → +Y
- Blue → +Z

**Point Size** — slider (1–15). Adjusts rendered point size in the viewport.

**Color Mode** — dropdown with three options:

| Mode | Description |
|---|---|
| RGB | Per-point colours from the COLMAP reconstruction |
| Elevation | Height-mapped using the Viridis colormap |
| Cyan | Flat cyan — useful for inspecting geometry without colour bias |

---

### Cropping

The crop panel trims the visible (and exported) point cloud along any axis.

**Axis Selector** — choose X Axis, Y Axis or Z Axis. The histogram and MIN/MAX spinboxes always show the selected axis.

**MIN / MAX spinboxes** — type values directly or drag the linked sliders. The viewport updates immediately on change. The histogram shows vertical lines:
- **Red** = current MIN boundary
- **Green** = current MAX boundary

**Auto99** — sets MIN/MAX to the 0.5th and 99.5th percentile of the selected axis, removing outlier points.

**Auto95** — sets MIN/MAX to the 2.5th and 97.5th percentile for a tighter crop.

**Reset Crop** — restores MIN/MAX to the full data bounds for the selected axis.

**Log Mode** — applies a symmetric log₁₀ transform to the histogram display. Useful when data has a wide dynamic range or heavy outliers. Crop boundaries are transformed accordingly.

> Crop values are stored per-axis in `current_crop = [xmin, xmax, ymin, ymax, zmin, zmax]` in display space. The crop is applied to both the viewport and the exported file.

---

### Histogram Panel

An embedded PyVista chart below the main viewport showing the distribution of points along the currently selected axis (100 bins).

- **Left-click** the histogram to set the MIN crop boundary at that data value.
- **Right-click** the histogram to set the MAX crop boundary at that data value.

Bar colours match the axis: **red** for X, **green** for Y, **blue** for Z.

---

### Bake SRT

**Button:** `✅ Bake SRT` (green = clean / red = unbaked changes pending)

Commits the current SRT transform permanently:

1. Applies the display-space 4×4 matrix to `cloud_poly` (the PyVista mesh).
2. Updates every `points3D[pid]['xyz']` in COLMAP space using the coordinate-frame-corrected `(S, R_c, T_c)`.
3. Updates every `images[iid]['qvec']` and `images[iid]['tvec']` with the correct camera pose transform (see [Coordinate System Notes](#coordinate-system-notes)).
4. Resets all SRT sliders to identity (Scale=1, everything else=0).
5. Writes a `bake` entry to `logfile.json`.

After baking, subsequent SRT adjustments and exports start from the new baked state. **You can bake multiple times** — each bake accumulates on top of the previous one.

> Always bake before exporting if you want the camera poses to match the point cloud.

---

### Export COLMAP

**Button:** `💾 Export COLMAP` + **format selector** `[BIN | TXT]`

Exports the transformed, cropped scene to a folder you choose.

**Files written:**

| Format | Files |
|---|---|
| BIN | `points3D.bin`, `images.bin`, `cameras.bin` |
| TXT | `points3D.txt`, `images.txt`, `cameras.txt` |

**What gets transformed:**
- Point XYZ coordinates — full SRT applied in COLMAP coordinate space, crop filter applied.
- Camera poses — rotation and translation updated consistently with the point transform.
- Camera intrinsics — written unchanged (scale does not affect focal length or principal point).

**What is preserved verbatim:**
- Track data (image_id / point2d_idx pairs) in `points3D`
- 2D observation data in `images`
- All camera model parameters

After export, a `export` entry is written to `logfile.json` including the output folder, format, and point counts.

---

### Settings

Three buttons in the **System** group:

| Button | Action |
|---|---|
| **Read** | Loads `settings.json` from the script directory — restores SRT, FOV, and crop bounds |
| **Write** | Saves current SRT, FOV, and crop bounds to `settings.json` |
| **N++** | Opens `settings.json` in Notepad++ (falls back to the system default editor if N++ is not installed) |

---

## File Reference

### settings.json

Located next to the script. Stores the last written state.

```json
{
  "srt": {
    "Scale":  1.0,
    "RotX":   0.0,
    "RotY":   0.0,
    "RotZ":   0.0,
    "TransX": 0.0,
    "TransY": 0.0,
    "TransZ": 0.0
  },
  "fov": 45.0,
  "crop": [-100.0, 100.0, -100.0, 100.0, -100.0, 100.0]
}
```

`crop` is a flat array: `[xmin, xmax, ymin, ymax, zmin, zmax]` in display space.

---

### logfile.json

Located next to the script. Appended on every Bake and Export — never overwritten, so you have a full history of operations.

**Bake entry example:**
```json
{
  "timestamp":     "2026-04-27T18:45:12",
  "event":         "bake",
  "points_total":  142857,
  "cameras_total": 48,
  "images_total":  48,
  "fov":           45.0,
  "srt": {
    "Scale":  1.0,
    "RotX":   0.0,
    "RotY":   90.0,
    "RotZ":   0.0,
    "TransX": -2.341,
    "TransY": 0.0,
    "TransZ": 1.05
  },
  "crop": {
    "X": { "min": -10.0, "max": 10.0 },
    "Y": { "min": -5.0,  "max": 5.0  },
    "Z": { "min": -8.0,  "max": 8.0  }
  }
}
```

**Export entry adds:**
```json
{
  "event":           "export",
  "output_folder":   "B:\\output\\gaussian",
  "export_fmt":      "BIN",
  "points_total":    142857,
  "points_exported": 98432,
  "points_cropped":  44425
}
```

---

## Coordinate System Notes

COLMAP uses a right-handed coordinate system where the camera looks along **+Z** with **-Y up** (OpenCV convention).

This tool displays the point cloud with **Y negated** (`xyz * [1, -1, 1]`) so the scene appears right-side up in the PyVista viewport. All SRT slider values are in this **display space**.

When exporting, the tool converts back to COLMAP space using the conjugation `R_colmap = F @ R_display @ F` and `T_colmap = F @ T_display` where `F = diag(1, -1, 1)`.

Camera pose transforms use the mathematically correct update:
```
C_new    = S * (R_c @ C) + T_c          # transform camera centre
R_cw_new = R_cw @ R_c.T                 # update rotation
t_cw_new = -R_cw_new @ C_new            # recompute translation
```
where `C = -R_cw.T @ t_cw` is the camera centre in world space.

---

## Supported Camera Models

| Model ID | Name | Params | Description |
|---|---|---|---|
| 0 | SIMPLE_PINHOLE | 3 | f, cx, cy |
| 1 | PINHOLE | 4 | fx, fy, cx, cy |
| 2 | SIMPLE_RADIAL | 4 | f, cx, cy, k |
| 3 | RADIAL | 5 | f, cx, cy, k1, k2 |
| 4 | OPENCV | 8 | fx, fy, cx, cy, k1, k2, p1, p2 |
| 5 | OPENCV_FISHEYE | 8 | fx, fy, cx, cy, k1, k2, k3, k4 |
| 6 | FULL_OPENCV | 12 | fx, fy, cx, cy, k1–k6, p1, p2 |

---

## Workflow Examples

### Level a scene to a known floor plane

1. Import project.
2. Click **Floor (3Pt)** and pick three points on the floor.
3. Verify the scene is level in the V05 (top-down) view.
4. Click **XZ Origin**, pick a reference point to centre the scene.
5. Click **✅ Bake SRT**.
6. Crop any ceiling clutter using **Y Axis** + **Auto99**.
7. Select **BIN** and click **💾 Export COLMAP**.

---

### Align a corridor to the Z axis

1. Import project.
2. Click **Y-Align (2Pt)** and pick two points along the corridor floor centre line.
3. Fine-tune with **RotY** slider if needed.
4. Bake, then export.

---

### Remove outlier points before Gaussian Splatting

1. Import project.
2. Set Color Mode → **Elevation** to spot outliers easily.
3. Select each axis in turn and use **Auto99** to auto-crop.
4. Fine-tune MIN/MAX by clicking directly on the histogram.
5. Export — the crop is applied automatically; `logfile.json` records how many points were removed.

---

### Iterative refinement workflow

1. Import → coarse SRT → Bake → Write settings.
2. Re-open next session → Read settings to restore last state.
3. Apply fine corrections → Bake again.
4. Export when satisfied. `logfile.json` records every bake step for reproducibility.

---

## Known Limitations

- **TXT loader** reads `points3D.txt` only — `images.txt` and `cameras.txt` are not parsed in text mode. Export of camera data in TXT mode still works if binary files were loaded first.
- **N++ button** is hardcoded to `C:\Program Files\Notepad++\notepad++.exe` and falls back to the OS default editor if not found.
- **Track filtering** — when cropping removes a point, its track entries in `images` are not back-purged. The exported `images.bin/txt` will still reference point IDs that no longer exist in `points3D`. This is harmless for visualisation in most tools but may cause COLMAP to report orphaned observations.
- **Scale affects translation units** — if you scale the scene, any subsequent translation values are in the scaled unit space. Apply translation before scale, or bake scale first.