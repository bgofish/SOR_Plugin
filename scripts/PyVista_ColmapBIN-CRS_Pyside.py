# COLMAP Point Editor v0.2.1
#================================================
import sys, os, struct, json, subprocess
import numpy as np

# QApplication must exist before any Qt widgets are created.
# Import only the minimal Qt surface needed for the splash first —
# the heavy libs (pyvista, vtk, scipy) are imported AFTER the splash shows.
from PySide6.QtWidgets import (QApplication, QSplashScreen, QMainWindow, QVBoxLayout, QHBoxLayout,
                              QPushButton, QWidget, QFileDialog, QSlider, QLabel, QLineEdit,
                              QComboBox, QGroupBox, QDoubleSpinBox, QRadioButton, QGridLayout, QButtonGroup)
from PySide6.QtGui import QKeySequence, QShortcut, QPixmap, QColor, QFont, QPainter
from PySide6.QtCore import Qt
_app = QApplication.instance() or QApplication(sys.argv)

# ── Show splash immediately, before any heavy imports ────────────────────────
def _show_splash():
    try:
        pix = QPixmap(540, 120)
        pix.fill(QColor("#1a1a2e"))
        p = QPainter(pix)
        p.setPen(QColor("#00d4ff"))
        p.setFont(QFont("Segoe UI", 18, QFont.Bold))
        p.drawText(pix.rect(), Qt.AlignCenter, "COLMAP Point Editor")
        p.setPen(QColor("#aaaaaa"))
        p.setFont(QFont("Segoe UI", 9))
        from PySide6.QtCore import QRect
        p.drawText(QRect(0, 75, 540, 35), Qt.AlignCenter, "Loading — this may take a moment…")
        p.end()
        splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
        splash.show()
        _app.processEvents()   # force the splash to paint immediately
        return splash
    except Exception:
        return None

_splash = _show_splash()

# ── Heavy imports (slow — VTK/PyVista/SciPy) ─────────────────────────────────
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial.transform import Rotation as Rot
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.multi_samples = 0

# --- Session guard via named mutex (Win32) or module-level flag (other platforms) ---
# The mutex is owned by the first process to create it and released automatically
# by the OS when that process exits — no stale state possible.
import tempfile

_MUTEX_NAME   = "COLMAPEditor_SessionMutex"
_mutex_handle = None   # holds the Win32 mutex handle for the lifetime of this process

def _session_already_launched() -> bool:
    global _mutex_handle
    if sys.platform != "win32":
        # Non-Windows: fall back to a module-level flag (good enough for one session)
        return _mutex_handle is not None

    import ctypes
    ERROR_ALREADY_EXISTS = 183
    handle = ctypes.windll.kernel32.CreateMutexW(None, True, _MUTEX_NAME)
    if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        # Mutex already owned by another (or same) process this session
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
        return True
    # We own the mutex — store handle so it stays alive (and locked) for this process
    _mutex_handle = handle
    return False

# Hide the PowerShell console window on Win32
_SUBPROCESS_FLAGS = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW

def _notify_already_running() -> None:
    """Show a Windows balloon notification warning that the editor is already open.
    Uses the Win32 NotifyIcon API via System.Windows.Forms (no extra packages).
    Silently skipped on non-Win32 platforms.
    """
    if sys.platform != "win32":
        return
    title_ps   = "COLMAP Point Editor"
    message_ps = "COLMAP Editor is already open. Please close the existing window before opening a new one."
    title_ps   = title_ps.replace("'", "\u2019")
    message_ps = message_ps.replace("'", "\u2019")
    ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Warning
$notify.BalloonTipIcon  = [System.Windows.Forms.ToolTipIcon]::Warning
$notify.BalloonTipTitle = '{title_ps}'
$notify.BalloonTipText  = '{message_ps}'
$notify.Visible = $true
$notify.ShowBalloonTip(8000)
Start-Sleep -Milliseconds 9000
$notify.Dispose()
"""
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
            creationflags=_SUBPROCESS_FLAGS,
        )
    except Exception:
        pass

# --- HELPERS ---
class CustomSlider(QSlider):
    def __init__(self, orientation, parent=None, default=0):
        super().__init__(orientation, parent); self.default_val = default
    def mouseDoubleClickEvent(self, event): self.setValue(self.default_val)

def qvec2rotmat(qvec): return Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
def rotmat2qvec(R):
    q = Rot.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]]) # w, x, y, z

class COLMAPProject:
    def __init__(self): self.cameras, self.images, self.points3D, self.is_bin = {}, {}, {}, True
    def load(self, folder):
        self.is_bin = os.path.exists(os.path.join(folder, "points3D.bin"))
        if self.is_bin: self._load_bin(folder)
        else: self._load_txt(folder)

    def _load_bin(self, folder):
        with open(os.path.join(folder, "points3D.bin"), "rb") as f:
            num = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num):
                # 43 byte header: Q(8) + ddd(24) + BBB(3) + d(8)
                d = struct.unpack("<QdddBBBd", f.read(43))
                t_len = struct.unpack("<Q", f.read(8))[0]
                tracks = f.read(t_len * 8)
                self.points3D[d[0]] = {'xyz': np.array(d[1:4]), 'rgb': np.array(d[4:7]), 'err': d[7], 'tracks': tracks}
        
        i_path = os.path.join(folder, "images.bin")
        if os.path.exists(i_path):
            with open(i_path, "rb") as f:
                num = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num):
                    h = struct.unpack("<IdddddddI", f.read(64))
                    name = ""
                    while True:
                        char = f.read(1)
                        if char == b"\x00" or not char: break
                        name += char.decode("utf-8")
                    num_p2d = struct.unpack("<Q", f.read(8))[0]
                    self.images[h[0]] = {'qvec': np.array(h[1:5]), 'tvec': np.array(h[5:8]), 'cam_id': h[8], 'name': name, 'p2d': f.read(num_p2d * 24)}

        c_path = os.path.join(folder, "cameras.bin")
        if os.path.exists(c_path):
            with open(c_path, "rb") as f:
                num = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num):
                    cid, model, w, h = struct.unpack("<IiQQ", f.read(24))
                    p_num = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:12}.get(model, 4)
                    params = struct.unpack(f"<{p_num}d", f.read(p_num * 8))
                    self.cameras[cid] = {'model': model, 'hw': (w,h), 'params': params}

    def _load_txt(self, folder):
        # --- points3D.txt ---
        p_path = os.path.join(folder, "points3D.txt")
        if os.path.exists(p_path):
            with open(p_path, "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip(): continue
                    p = line.split()
                    self.points3D[int(p[0])] = {
                        'xyz':    np.array(p[1:4], float),
                        'rgb':    np.array(p[4:7], int),
                        'err':    float(p[7]),
                        'tracks': p[8:]
                    }

        # --- cameras.txt ---
        MODEL_IDS = {"SIMPLE_PINHOLE":0,"PINHOLE":1,"SIMPLE_RADIAL":2,"RADIAL":3,
                     "OPENCV":4,"OPENCV_FISHEYE":5,"FULL_OPENCV":6}
        c_path = os.path.join(folder, "cameras.txt")
        if os.path.exists(c_path):
            with open(c_path, "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip(): continue
                    p = line.split()
                    if len(p) < 5: continue
                    cid    = int(p[0])
                    model  = MODEL_IDS.get(p[1], 1)
                    w, h   = int(p[2]), int(p[3])
                    params = tuple(float(x) for x in p[4:])
                    self.cameras[cid] = {'model': model, 'hw': (w, h), 'params': params}

        # --- images.txt ---
        i_path = os.path.join(folder, "images.txt")
        if os.path.exists(i_path):
            with open(i_path, "r") as f:
                lines = [l for l in f if not l.startswith("#") and l.strip()]
            idx = 0
            while idx < len(lines) - 1:
                p = lines[idx].split()
                if len(p) < 9: idx += 1; continue
                iid    = int(p[0])
                qvec   = np.array(p[1:5], float)
                tvec   = np.array(p[5:8], float)
                cam_id = int(p[8])
                name   = p[9] if len(p) > 9 else ""
                # Second line: X Y POINT3D_ID triplets — store as bytes to match bin format
                p2d_parts = lines[idx + 1].split()
                p2d_bytes = b""
                for j in range(0, len(p2d_parts) - 2, 3):
                    try:
                        x2   = float(p2d_parts[j])
                        y2   = float(p2d_parts[j + 1])
                        p3id = int(p2d_parts[j + 2])
                        p2d_bytes += struct.pack("<ddq", x2, y2, p3id)
                    except (ValueError, IndexError):
                        break
                self.images[iid] = {
                    'qvec':   qvec,
                    'tvec':   tvec,
                    'cam_id': cam_id,
                    'name':   name,
                    'p2d':    p2d_bytes
                }
                idx += 2

class COLMAPExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COLMAP Point Editor v0.2.0")
        self.resize(1750, 1050); self.proj, self.cloud_poly = None, None
        self.bounds = [0.0]*6; self.current_crop = [0.0]*6; self.bins = None
        self.picked_pts, self.pick_mode = [], None
        self._culled_ids = set()   # image IDs removed by cull
        _script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
        self.settings_file = os.path.join(_script_dir, "PyVista_Colmap_settings.json")

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); sidebar = QVBoxLayout()

        # IO & SYSTEM
        set_grp = QGroupBox("System"); h_set = QHBoxLayout()
        for t, f in [("Read", self.load_settings), ("Write", self.save_settings), ("Notepad++", self.open_in_npp)]:
            btn = QPushButton(t); btn.clicked.connect(f); h_set.addWidget(btn)
        p_v = QVBoxLayout(); p_v.addLayout(h_set)
        btn_load = QPushButton("📂 Import Project"); btn_load.clicked.connect(self.open_file)
        self.btn_bake = QPushButton("✅ SRT Baked"); self.btn_bake.clicked.connect(self.bake_srt)
        export_row = QHBoxLayout()
        btn_export = QPushButton("💾 Export COLMAP"); btn_export.clicked.connect(self.export_project)
        export_row.addWidget(btn_export)
        fmt_row = QHBoxLayout()
        self.radio_bin = QRadioButton("BIN"); self.radio_bin.setChecked(True)
        self.radio_txt = QRadioButton("TXT")
        fmt_grp = QButtonGroup(self); fmt_grp.addButton(self.radio_bin); fmt_grp.addButton(self.radio_txt)
        fmt_row.addWidget(QLabel("Format:")); fmt_row.addWidget(self.radio_bin); fmt_row.addWidget(self.radio_txt); fmt_row.addStretch()
        p_v.addWidget(btn_load); p_v.addWidget(self.btn_bake); p_v.addLayout(export_row); p_v.addLayout(fmt_row)
        set_grp.setLayout(p_v); sidebar.addWidget(set_grp)

        # VIEW & ALIGN
        v_grp = QGroupBox("View & Alignment"); v_v = QVBoxLayout(); grid = QGridLayout()
        view_labels = ["NW", "BACK", "NE", "LEFT", "TOP", "RIGHT", "SW", "FRONT", "SE"]
        numpad_keys = [Qt.Key_7, Qt.Key_8, Qt.Key_9,
                       Qt.Key_4, Qt.Key_5, Qt.Key_6,
                       Qt.Key_1, Qt.Key_2, Qt.Key_3]
        numpad_display = [7, 8, 9, 4, 5, 6, 1, 2, 3]
        for i, (label, key, num) in enumerate(zip(view_labels, numpad_keys, numpad_display)):
            btn = QPushButton(f"{label}\n[Num {num}]")
            btn.clicked.connect(lambda c, idx=i: self.apply_view_preset(idx))
            btn.setToolTip(f"Numpad {num}")
            grid.addWidget(btn, i//3, i%3)
            sc = QShortcut(QKeySequence(key | Qt.KeypadModifier), self)
            sc.activated.connect(lambda idx=i: self.apply_view_preset(idx))
        v_v.addLayout(grid)
        self.spin_fov = QDoubleSpinBox(); self.spin_fov.setRange(10, 120); self.spin_fov.setValue(45)
        self.spin_fov.valueChanged.connect(self.update_preview)
        fov_row = QHBoxLayout()
        fov_row.addWidget(QLabel("FOV:")); fov_row.addWidget(self.spin_fov)
        self.btn_ortho = QPushButton("ORTHO"); self.btn_ortho.setCheckable(True); self.btn_ortho.setChecked(False)
        self.btn_ortho.setStyleSheet("QPushButton:checked { background-color: #1e90ff; color: white; font-weight: bold; }")
        self.btn_ortho.toggled.connect(self._toggle_ortho)
        fov_row.addWidget(self.btn_ortho)
        v_v.addLayout(fov_row)
        
        self.pick_btn_map = {}
        for t, m in [("Floor (3Pt)", 'plane'), ("Y-Align (2Pt)", 'align'), ("XYZ Origin", 'xyz_origin'), ("XZ Origin", 'xz_origin')]:
            btn = QPushButton(t); btn.setCheckable(True); btn.clicked.connect(lambda c, mode=m: self.start_pick(mode))
            v_v.addWidget(btn); self.pick_btn_map[m] = btn
        v_grp.setLayout(v_v); sidebar.addWidget(v_grp)

        # SRT
        srt_grp = QGroupBox("Global SRT"); srt_v = QVBoxLayout(); self.srt_ctrl = {}
        cfgs = [('Scale',1.0,0.01,20),('RotX',0.0,-180,180),('RotY',0.0,-180,180),('RotZ',0.0,-180,180),('TransX',0.0,-5000,5000),('TransY',0.0,-5000,5000),('TransZ',0.0,-5000,5000)]
        for l, s, r1, r2 in cfgs:
            srt_v.addWidget(QLabel(f"<b>{l}</b>"))
            h = QHBoxLayout(); sp = QDoubleSpinBox(); sp.setRange(r1, r2); sp.setValue(s); sp.setDecimals(3)
            sl = CustomSlider(Qt.Horizontal, default=int(s*100) if l!='Scale' else 0); sl.setRange(int(r1*100), int(r2*100)); sl.setValue(int(s*100))
            sp.valueChanged.connect(lambda v, s_obj=sl: s_obj.blockSignals(True) or s_obj.setValue(int(v*100)) or s_obj.blockSignals(False))
            sl.valueChanged.connect(lambda v, p_obj=sp: p_obj.blockSignals(True) or p_obj.setValue(v/100.0) or p_obj.blockSignals(False))
            sp.valueChanged.connect(self.on_srt_change)
            sl.valueChanged.connect(self.on_srt_change); h.addWidget(sp); h.addWidget(sl); srt_v.addLayout(h); self.srt_ctrl[l] = sp
        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(0.001, 1.0); self.spin_step.setValue(0.02)
        self.spin_step.valueChanged.connect(self.update_step_sizes); srt_v.addWidget(QLabel("Step Size:")); srt_v.addWidget(self.spin_step); srt_grp.setLayout(srt_v); sidebar.addWidget(srt_grp)

        # VISUALS (points only)
        t_grp = QGroupBox("Visuals"); t_v = QVBoxLayout()
        tog_row = QHBoxLayout()
        self.btn_origin_vis = QPushButton("[Axis]"); self.btn_origin_vis.setCheckable(True); self.btn_origin_vis.setChecked(True); self.btn_origin_vis.toggled.connect(self.update_preview)
        self.btn_cam_vis = QPushButton("[Cameras]"); self.btn_cam_vis.setCheckable(True); self.btn_cam_vis.setChecked(False); self.btn_cam_vis.toggled.connect(self.update_preview)
        tog_row.addWidget(self.btn_origin_vis); tog_row.addWidget(self.btn_cam_vis)
        t_v.addLayout(tog_row)
        pts_row = QHBoxLayout()
        self.sld_pts = QSlider(Qt.Horizontal); self.sld_pts.setRange(1, 15); self.sld_pts.setValue(3); self.sld_pts.valueChanged.connect(self.update_preview)
        pts_row.addWidget(QLabel("Pt Size:")); pts_row.addWidget(self.sld_pts)
        cam_sz_row = QHBoxLayout()
        self.sld_cam_size = QSlider(Qt.Horizontal); self.sld_cam_size.setRange(1, 200); self.sld_cam_size.setValue(20); self.sld_cam_size.valueChanged.connect(self.update_preview)
        cam_sz_row.addWidget(QLabel("Cam Size:")); cam_sz_row.addWidget(self.sld_cam_size)
        t_v.addLayout(pts_row); t_v.addLayout(cam_sz_row)
        self.combo_color = QComboBox(); self.combo_color.addItems(["RGB", "Elevation", "Cyan"]); self.combo_color.currentIndexChanged.connect(self.update_preview); t_v.addWidget(self.combo_color)
        t_grp.setLayout(t_v); sidebar.addWidget(t_grp)

        # CAMERAS — cull controls in their own group
        cam_grp = QGroupBox("Camera Culling"); cam_v = QVBoxLayout()

        # Random % cull — slider + buttons on one row
        cull_row = QHBoxLayout()
        self.sld_cull = QSlider(Qt.Horizontal); self.sld_cull.setRange(5, 75); self.sld_cull.setValue(20)
        self.lbl_cull = QLabel("Cull 20%"); self.lbl_cull.setFixedWidth(58)
        self.sld_cull.valueChanged.connect(lambda v: self.lbl_cull.setText(f"Cull {v}%"))
        btn_cull_apply = QPushButton("Rnd Roll"); btn_cull_apply.clicked.connect(self.roll_cull)
        btn_cull_clear = QPushButton("Clear All"); btn_cull_clear.clicked.connect(self.clear_cull)
        cull_row.addWidget(self.lbl_cull); cull_row.addWidget(self.sld_cull); cull_row.addWidget(btn_cull_apply); cull_row.addWidget(btn_cull_clear)
        cam_v.addLayout(cull_row)

        # Name-based cull
        name_row = QHBoxLayout()
        self.txt_cull_name = QLineEdit(); self.txt_cull_name.setPlaceholderText("*back*   012345   012340-012350")
        btn_cull_name = QPushButton("Cull Match"); btn_cull_name.clicked.connect(self.cull_by_name)
        name_row.addWidget(self.txt_cull_name); name_row.addWidget(btn_cull_name)
        cam_v.addLayout(name_row)
        self.lbl_cull_name_result = QLabel("")
        cam_v.addWidget(self.lbl_cull_name_result)

        # Nth / odd / even cull — all on one row
        nth_row = QHBoxLayout()
        self.spin_nth = QDoubleSpinBox()
        self.spin_nth.setRange(2, 100); self.spin_nth.setDecimals(0); self.spin_nth.setValue(2); self.spin_nth.setFixedWidth(58)
        btn_nth  = QPushButton("Every Nth"); btn_nth.clicked.connect(self.cull_nth)
        btn_odd  = QPushButton("Odd");       btn_odd.clicked.connect(self.cull_odd)
        btn_even = QPushButton("Even");      btn_even.clicked.connect(self.cull_even)
        nth_row.addWidget(self.spin_nth); nth_row.addWidget(btn_nth); nth_row.addWidget(btn_odd); nth_row.addWidget(btn_even)
        cam_v.addLayout(nth_row)
        self.lbl_nth_result = QLabel("")
        cam_v.addWidget(self.lbl_nth_result)
        cam_grp.setLayout(cam_v); sidebar.addWidget(cam_grp)
        sidebar.addStretch(); main_layout.addLayout(sidebar, 1)

        # VIEWPORT & CROP
        content = QVBoxLayout()
        self.plotter = QtInteractor(None); self.hist_plotter = QtInteractor(None); self.hist_plotter.setMaximumHeight(160)
        content.addWidget(self.plotter, 6); content.addWidget(self.hist_plotter, 2)
        c_grp = QGroupBox("Cropping"); c_v = QVBoxLayout(); row = QHBoxLayout()
        self.axis_sel = QComboBox(); self.axis_sel.addItems(["X Axis", "Y Axis", "Z Axis"]); self.axis_sel.setCurrentIndex(1); self.axis_sel.currentIndexChanged.connect(self.sync_crop_ui)
        btn99, btn95 = QPushButton("Auto99"), QPushButton("Auto95")
        btn99.clicked.connect(lambda: self.auto_crop_stat(0.99)); btn95.clicked.connect(lambda: self.auto_crop_stat(0.95))
        self.rad_log = QRadioButton("Log Mode"); self.rad_log.toggled.connect(self.update_preview)
        row.addWidget(self.axis_sel); row.addStretch(); row.addWidget(btn99); row.addWidget(btn95); row.addWidget(self.rad_log)
        btn_res = QPushButton("Reset Crop"); btn_res.clicked.connect(self.reset_crop); row.addWidget(btn_res); c_v.addLayout(row)
        self.crop_ui = {}
        for l in ["min", "max"]:
            r = QHBoxLayout(); sp = QDoubleSpinBox(); sp.setRange(-1e7, 1e7); sl = CustomSlider(Qt.Horizontal); sl.setRange(0, 1000)
            sp.valueChanged.connect(self.update_preview); sl.valueChanged.connect(self.on_slider_move)
            r.addWidget(QLabel(l.upper())); r.addWidget(sp); r.addWidget(sl); c_v.addLayout(r); self.crop_ui[l] = sp; self.crop_ui[f"{l}_sl"] = sl
        c_grp.setLayout(c_v); content.addWidget(c_grp); main_layout.addLayout(content, 4)
        self.hist_plotter.iren.add_observer("LeftButtonPressEvent", lambda o,e: self.on_hist_click(o,e,"min"))
        self.hist_plotter.iren.add_observer("RightButtonPressEvent", lambda o,e: self.on_hist_click(o,e,"max"))
        self.plotter.set_background("black"); self.set_bake_state(False)

    def get_mat(self):
        S, R = self.srt_ctrl['Scale'].value(), Rot.from_euler('xyz', [self.srt_ctrl['RotX'].value(), self.srt_ctrl['RotY'].value(), self.srt_ctrl['RotZ'].value()], degrees=True).as_matrix()
        m = np.eye(4); m[:3,:3] = R * S; m[:3,3] = [self.srt_ctrl['TransX'].value(), self.srt_ctrl['TransY'].value(), self.srt_ctrl['TransZ'].value()]
        return m

    def get_colmap_srt(self):
        """Return (S, R_c, T_c) — the SRT expressed in original COLMAP coordinate space."""
        S = self.srt_ctrl['Scale'].value()
        R_d = Rot.from_euler('xyz', [self.srt_ctrl['RotX'].value(),
                                     self.srt_ctrl['RotY'].value(),
                                     self.srt_ctrl['RotZ'].value()], degrees=True).as_matrix()
        T_d = np.array([self.srt_ctrl['TransX'].value(),
                        self.srt_ctrl['TransY'].value(),
                        self.srt_ctrl['TransZ'].value()])
        # Display space has X and Y negated: P_display = F @ P_colmap, F = diag(-1,-1,1)
        # A display-space rotation R_d corresponds to F @ R_d @ F in COLMAP space.
        # Translation is purely additive in display space, so T_colmap = F @ T_d.
        F = np.diag([-1.0, -1.0, 1.0])
        return S, F @ R_d @ F, F @ T_d
        S, R = self.srt_ctrl['Scale'].value(), Rot.from_euler('xyz', [self.srt_ctrl['RotX'].value(), self.srt_ctrl['RotY'].value(), self.srt_ctrl['RotZ'].value()], degrees=True).as_matrix()
        m = np.eye(4); m[:3,:3] = R * S; m[:3,3] = [self.srt_ctrl['TransX'].value(), self.srt_ctrl['TransY'].value(), self.srt_ctrl['TransZ'].value()]
        return m

    def _make_frustum(self, C, R_cw, scale, aspect=1.333, fov_deg=60.0):
        """
        Build a camera frustum mesh in display space.

        C      : camera centre in display space (3,)
        R_cw   : rotation matrix camera-from-world in display space (3,3)
              (columns are the world axes expressed in camera space)
        scale  : half-depth of the frustum
        Returns a pv.PolyData wireframe.
        """
        # Camera axes in world/display space: columns of R_cw.T
        R_wc = R_cw.T          # world-from-camera
        forward = R_wc[:, 2]   # +Z in camera space = optical axis
        right   = R_wc[:, 0]   # +X
        up      = R_wc[:, 1]   # +Y  (will negate for frustum corners so they fan out)

        half_h = scale * np.tan(np.radians(fov_deg / 2.0))
        half_w = half_h * aspect
        depth  = scale

        # Four corners of the far plane, tip at C
        tip = C
        tl = C + depth * forward - half_w * right + half_h * up
        tr = C + depth * forward + half_w * right + half_h * up
        bl = C + depth * forward - half_w * right - half_h * up
        br = C + depth * forward + half_w * right - half_h * up

        pts = np.array([tip, tl, tr, br, bl], dtype=np.float32)

        # Lines: 4 edges from tip + 4 edges of the rectangle
        lines = np.array([
            2, 0, 1,
            2, 0, 2,
            2, 0, 3,
            2, 0, 4,
            2, 1, 2,
            2, 2, 3,
            2, 3, 4,
            2, 4, 1,
        ])
        mesh = pv.PolyData()
        mesh.points = pts
        mesh.lines  = lines
        return mesh

    def _sorted_image_ids(self):
        """Return image IDs sorted by filename — consistent basis for nth/odd/even."""
        if not self.proj or not self.proj.images:
            return []
        return [iid for iid, _ in sorted(self.proj.images.items(),
                                          key=lambda kv: kv[1]['name'])]

    def cull_nth(self):
        """Cull every Nth image (0-based index: 0, N, 2N, …)."""
        n = max(2, int(self.spin_nth.value()))
        ids = self._sorted_image_ids()
        matched = {iid for i, iid in enumerate(ids) if i % n == 0}
        self._culled_ids |= matched
        self.lbl_nth_result.setText(f"Every {n}th: {len(matched)} culled  ({len(self._culled_ids)} total)")
        self.btn_cam_vis.setChecked(True); self.update_preview()

    def cull_odd(self):
        """Cull images at odd positions (1-based: 1, 3, 5, …)."""
        ids = self._sorted_image_ids()
        matched = {iid for i, iid in enumerate(ids) if i % 2 == 1}
        self._culled_ids |= matched
        self.lbl_nth_result.setText(f"Odd: {len(matched)} culled  ({len(self._culled_ids)} total)")
        self.btn_cam_vis.setChecked(True); self.update_preview()

    def cull_even(self):
        """Cull images at even positions (1-based: 2, 4, 6, …)."""
        ids = self._sorted_image_ids()
        matched = {iid for i, iid in enumerate(ids) if i % 2 == 0}
        self._culled_ids |= matched
        self.lbl_nth_result.setText(f"Even: {len(matched)} culled  ({len(self._culled_ids)} total)")
        self.btn_cam_vis.setChecked(True); self.update_preview()

    def cull_by_name(self):
        """Cull cameras whose filename matches a wildcard, suffix number, range, or substring."""
        import fnmatch, re
        if not self.proj or not self.proj.images:
            return
        pattern = self.txt_cull_name.text().strip()
        if not pattern:
            self.lbl_cull_name_result.setText("Enter a pattern first.")
            return

        # Detect numeric range  e.g.  012340-012350
        range_match = re.fullmatch(r'(\d+)\s*-\s*(\d+)', pattern)

        matched = set()
        for iid, img in self.proj.images.items():
            name = img['name']
            stem = os.path.splitext(name)[0]
            after_us = stem.rsplit('_', 1)[-1]   # numeric suffix after last '_'

            if range_match:
                # Numeric range — compare as integers so 012340 == 12340
                lo, hi = int(range_match.group(1)), int(range_match.group(2))
                if re.fullmatch(r'\d+', after_us) and lo <= int(after_us) <= hi:
                    matched.add(iid)

            elif '*' in pattern or '?' in pattern:
                if fnmatch.fnmatch(name.lower(), pattern.lower()) or \
                   fnmatch.fnmatch(stem.lower(), pattern.lower()):
                    matched.add(iid)

            elif re.fullmatch(r'\d+', pattern):
                # Exact numeric suffix
                if after_us == pattern or int(after_us or -1) == int(pattern):
                    matched.add(iid)

            else:
                # Plain substring
                if pattern.lower() in name.lower():
                    matched.add(iid)

        self._culled_ids |= matched
        n = len(matched)
        total_culled = len(self._culled_ids)
        self.lbl_cull_name_result.setText(
            f"{n} matched, {total_culled} total culled" if n else "No matches found."
        )
        if n:
            self.btn_cam_vis.setChecked(True)
            self.update_preview()

    def roll_cull(self):
        """Randomly select a percentage of cameras to cull."""
        if not self.proj or not self.proj.images:
            return
        import random
        pct   = self.sld_cull.value() / 100.0
        ids   = list(self.proj.images.keys())
        n     = max(1, int(round(len(ids) * pct)))
        self._culled_ids = set(random.sample(ids, n))
        kept  = len(ids) - len(self._culled_ids)
        self.lbl_cull.setText(f"Cull {self.sld_cull.value()}%  ({len(self._culled_ids)} removed, {kept} kept)")
        self.btn_cam_vis.setChecked(True)
        self.update_preview()

    def clear_cull(self):
        """Restore all culled cameras."""
        self._culled_ids = set()
        self.lbl_cull.setText(f"Cull {self.sld_cull.value()}%")
        self.update_preview()

    def draw_cameras(self):
        """Render all camera frustums into the plotter — culled ones in red."""
        self.plotter.remove_actor("cameras_kept")
        self.plotter.remove_actor("cameras_culled")
        if not self.btn_cam_vis.isChecked() or not self.proj or not self.proj.images:
            return

        scale  = self.sld_cam_size.value() * 0.01
        M      = self.get_mat()
        R_srt  = M[:3, :3]
        T_srt  = M[:3,  3]
        F      = np.diag([-1.0, -1.0, 1.0])

        kept_blocks   = pv.MultiBlock()
        culled_blocks = pv.MultiBlock()

        for iid, img in self.proj.images.items():
            R_cw_col  = qvec2rotmat(img['qvec'])
            t_cw_col  = np.array(img['tvec'], float)
            C_col     = -R_cw_col.T @ t_cw_col
            C_disp    = R_srt @ (F @ C_col) + T_srt
            R_cw_disp = R_cw_col @ F.T @ R_srt.T
            mesh      = self._make_frustum(C_disp, R_cw_disp, scale)
            if iid in self._culled_ids:
                culled_blocks.append(mesh)
            else:
                kept_blocks.append(mesh)

        if len(kept_blocks):
            self.plotter.add_mesh(kept_blocks.combine(), name="cameras_kept",
                                  color="yellow", line_width=1, reset_camera=False)
        if len(culled_blocks):
            self.plotter.add_mesh(culled_blocks.combine(), name="cameras_culled",
                                  color="red", line_width=1, reset_camera=False)

    def _toggle_ortho(self, checked: bool):
        self.plotter.camera.parallel_projection = checked
        self.btn_ortho.setText("ORTHO" if checked else "PERSP")
        self.spin_fov.setEnabled(not checked)
        self.plotter.render()

    def update_preview(self):
        if not self.cloud_poly: return
        self.plotter.camera.view_angle = self.spin_fov.value()
        idx = self.axis_sel.currentIndex(); self.current_crop[2*idx], self.current_crop[2*idx+1] = self.crop_ui['min'].value(), self.crop_ui['max'].value()
        temp = self.cloud_poly.copy().transform(self.get_mat(), inplace=True); c = self.current_crop
        mask = (temp.points[:,0]>=c[0])&(temp.points[:,0]<=c[1])&(temp.points[:,1]>=c[2])&(temp.points[:,1]<=c[3])&(temp.points[:,2]>=c[4])&(temp.points[:,2]<=c[5])
        active = temp.extract_points(mask) if np.any(mask) else None
        if active:
            cm = self.combo_color.currentIndex()
            if cm == 1: self.plotter.add_mesh(active, name="cloud", scalars=active.points[:,1], cmap="viridis", show_scalar_bar=False, reset_camera=False, point_size=self.sld_pts.value())
            elif cm == 2: self.plotter.add_mesh(active, name="cloud", color="cyan", reset_camera=False, point_size=self.sld_pts.value())
            else: self.plotter.add_mesh(active, name="cloud", scalars="rgb", rgb=True, reset_camera=False, point_size=self.sld_pts.value())
        else: self.plotter.remove_actor("cloud")
        self.draw_cameras()
        self.update_histogram(temp.points); self.toggle_origin(self.btn_origin_vis.isChecked()); self.plotter.render()

    def update_histogram(self, pts):
        self.hist_plotter.clear(); idx = self.axis_sel.currentIndex(); data = pts[:, idx]
        if self.rad_log.isChecked(): data = np.sign(data) * np.log10(np.abs(data) + 1)
        counts, self.bins = np.histogram(data, bins=100); chart = pv.Chart2D()
        chart.bar(self.bins[:-1], counts, color=["#FF4444","#44FF44","#4444FF"][idx])
        c_min, c_max = self.current_crop[2*idx], self.current_crop[2*idx+1]
        if self.rad_log.isChecked(): 
            c_min = np.sign(c_min)*np.log10(np.abs(c_min)+1); c_max = np.sign(c_max)*np.log10(np.abs(c_max)+1)
        chart.line([c_min, c_min], [0, counts.max()], color="red",   width=2)
        chart.line([c_max, c_max], [0, counts.max()], color="green", width=2)
        self.hist_plotter.add_chart(chart); self.hist_plotter.render()

    def save_settings(self):
        d = {
            "srt": {k: float(v.value()) for k, v in self.srt_ctrl.items()},
            "fov": float(self.spin_fov.value()),
            "crop": [float(x) for x in self.current_crop]
        }
        with open(self.settings_file, "w") as f: json.dump(d, f, indent=4)

    def load_settings(self):
        if not os.path.exists(self.settings_file): return
        with open(self.settings_file, "r") as f: d = json.load(f)
        for k, v in d["srt"].items(): self.srt_ctrl[k].setValue(v)
        self.spin_fov.setValue(d.get("fov", 45)); self.current_crop = d.get("crop", [0.0]*6); self.sync_crop_ui()

    def export_project(self):
        f = QFileDialog.getExistingDirectory(self, "Export Folder")
        if not (f and self.proj): return
        use_bin = self.radio_bin.isChecked()
        S, R_c, T_c = self.get_colmap_srt()

        def _transform_image(img):
            """Return (q_new, t_new) with correct COLMAP camera pose update."""
            R_cw = qvec2rotmat(img['qvec'])
            t_cw = np.array(img['tvec'])
            C        = -R_cw.T @ t_cw          # camera centre in world space
            C_new    = S * (R_c @ C) + T_c     # transform centre
            R_cw_new = R_cw @ R_c.T            # new rotation
            t_cw_new = -R_cw_new @ C_new       # new translation
            return rotmat2qvec(R_cw_new), t_cw_new

        def _point_xyz_new(p):
            return S * (R_c @ np.array(p['xyz'])) + T_c

        # Apply current crop bounds — crop is in display space (Y-flipped),
        # so convert each point to display space before testing.
        xmin, xmax, ymin, ymax, zmin, zmax = self.current_crop
        def _in_crop(p):
            x, y, z = p['xyz']           # COLMAP space
            dx, dy, dz = -x, -y, z      # display space (X and Y flipped)
            return (xmin <= dx <= xmax and
                    ymin <= dy <= ymax and
                    zmin <= dz <= zmax)

        cropped_points = {pid: p for pid, p in self.proj.points3D.items() if _in_crop(p)}
        total_pts   = len(self.proj.points3D)
        cropped_pts = len(cropped_points)
        print(f"[export] Crop: {cropped_pts:,} / {total_pts:,} points")

        # Filter culled cameras
        export_images  = {iid: img for iid, img in self.proj.images.items() if iid not in self._culled_ids}
        export_cam_ids = {img['cam_id'] for img in export_images.values()}
        export_cameras = {cid: cam for cid, cam in self.proj.cameras.items() if cid in export_cam_ids}
        print(f"[export] Cameras: {len(export_images)} / {len(self.proj.images)} images  ({len(self._culled_ids)} culled)")

        def _track_str(tracks):
            s = ""
            if isinstance(tracks, bytes):
                for j in range(len(tracks) // 8):
                    iid2, pt2d = struct.unpack_from("<ii", tracks, j * 8)
                    s += f" {iid2} {pt2d}"
            elif isinstance(tracks, list):
                s = " " + " ".join(str(t) for t in tracks)
            return s

        def _p2d_str(p2d):
            if not isinstance(p2d, bytes) or len(p2d) < 24:
                return ""
            parts = []
            for j in range(len(p2d) // 24):
                x2, y2 = struct.unpack_from("<dd", p2d, j * 24)
                p3id   = struct.unpack_from("<q",  p2d, j * 24 + 16)[0]
                parts.append(f"{x2:.4f} {y2:.4f} {p3id}")
            return " ".join(parts)

        MODEL_NAMES = {0:"SIMPLE_PINHOLE", 1:"PINHOLE", 2:"SIMPLE_RADIAL", 3:"RADIAL",
                       4:"OPENCV", 5:"OPENCV_FISHEYE", 6:"FULL_OPENCV"}

        if use_bin:
            # --- points3D.bin ---
            with open(os.path.join(f, "points3D.bin"), "wb") as file:
                file.write(struct.pack("<Q", len(cropped_points)))
                for pid, p in cropped_points.items():
                    xyz = _point_xyz_new(p)
                    r, g, b = int(p['rgb'][0]), int(p['rgb'][1]), int(p['rgb'][2])
                    file.write(struct.pack("<QdddBBBd", pid, *xyz, r, g, b, p['err']))
                    tracks = p['tracks']
                    if isinstance(tracks, bytes):
                        file.write(struct.pack("<Q", len(tracks) // 8))
                        file.write(tracks)
                    else:
                        file.write(struct.pack("<Q", 0))

            # --- images.bin ---
            with open(os.path.join(f, "images.bin"), "wb") as file:
                file.write(struct.pack("<Q", len(export_images)))
                for iid, img in export_images.items():
                    q_new, t_new = _transform_image(img)
                    file.write(struct.pack("<I",    iid))
                    file.write(struct.pack("<dddd", *q_new))
                    file.write(struct.pack("<ddd",  *t_new))
                    file.write(struct.pack("<I",    img['cam_id']))
                    file.write(img['name'].encode() + b"\x00")
                    p2d = img['p2d']
                    if isinstance(p2d, bytes):
                        file.write(struct.pack("<Q", len(p2d) // 24))
                        file.write(p2d)
                    else:
                        file.write(struct.pack("<Q", 0))

            # --- cameras.bin ---
            with open(os.path.join(f, "cameras.bin"), "wb") as file:
                file.write(struct.pack("<Q", len(export_cameras)))
                for cid, cam in export_cameras.items():
                    file.write(struct.pack("<IiQQ", cid, cam['model'], *cam['hw']))
                    file.write(struct.pack(f"<{len(cam['params'])}d", *cam['params']))

        else:
            # --- points3D.txt ---
            with open(os.path.join(f, "points3D.txt"), "w") as file:
                file.write("# 3D point list\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
                file.write(f"# Number of points: {len(cropped_points)}\n")
                for pid, p in cropped_points.items():
                    xyz = _point_xyz_new(p)
                    r, g, b = int(p['rgb'][0]), int(p['rgb'][1]), int(p['rgb'][2])
                    file.write(f"{pid} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {r} {g} {b} {p['err']:.6f}{_track_str(p['tracks'])}\n")

            # --- images.txt ---
            with open(os.path.join(f, "images.txt"), "w") as file:
                file.write("# Image list\n#   IMAGE_ID, QW,QX,QY,QZ, TX,TY,TZ, CAMERA_ID, NAME\n")
                file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
                file.write(f"# Number of images: {len(export_images)}\n")
                for iid, img in export_images.items():
                    q_new, t_new = _transform_image(img)
                    file.write(f"{iid} {q_new[0]:.9f} {q_new[1]:.9f} {q_new[2]:.9f} {q_new[3]:.9f} "
                               f"{t_new[0]:.9f} {t_new[1]:.9f} {t_new[2]:.9f} {img['cam_id']} {img['name']}\n")
                    file.write(_p2d_str(img['p2d']) + "\n")

            # --- cameras.txt ---
            with open(os.path.join(f, "cameras.txt"), "w") as file:
                file.write("# Camera list with one line of data per camera:\n")
                file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                file.write(f"# Number of cameras: {len(export_cameras)}\n")
                for cid, cam in export_cameras.items():
                    model_name = MODEL_NAMES.get(cam['model'], f"UNKNOWN_{cam['model']}")
                    params_str = " ".join(f"{p:.6f}" for p in cam['params'])
                    file.write(f"{cid} {model_name} {cam['hw'][0]} {cam['hw'][1]} {params_str}\n")

        fmt = "BIN" if use_bin else "TXT"
        print(f"Exported {fmt} to {f}")
        self._write_logfile(event="export", output_folder=f, export_fmt=fmt,
                            points_exported=cropped_pts, points_total=total_pts)

    def on_hist_click(self, o, e, mode):
        if self.bins is None: return
        x, _ = o.GetEventPosition(); pct = (x/self.hist_plotter.width() - 0.08)/0.84
        val = self.bins[int(np.clip(pct, 0, 1) * (len(self.bins)-1))]
        if self.rad_log.isChecked(): val = np.sign(val) * (10**np.abs(val) - 1)
        self.crop_ui[mode].setValue(val)

    def start_pick(self, m):
        self.plotter.disable_picking()
        for b in self.pick_btn_map.values(): b.setChecked(False)
        self.pick_btn_map[m].setChecked(True)
        self.pick_mode = m; self.picked_pts = []
        self._clear_pick_visuals()
        self.plotter.enable_point_picking(callback=self.pick_callback, show_message=True, color='yellow', point_size=12)

    def _clear_pick_visuals(self):
        for name in ['_pick_pts', '_pick_lines', '_pick_normal']:
            self.plotter.remove_actor(name)

    def _draw_pick_visuals(self):
        self._clear_pick_visuals()
        pts = self.picked_pts
        if not pts: return
        # Picked point spheres
        cloud = pv.PolyData(np.array(pts))
        self.plotter.add_mesh(cloud, name='_pick_pts', color='yellow', point_size=16, render_points_as_spheres=True, reset_camera=False)

        if self.pick_mode == 'align' and len(pts) >= 2:
            # Draw line between 2 points
            line = pv.Line(pts[0], pts[1])
            self.plotter.add_mesh(line, name='_pick_lines', color='cyan', line_width=3, reset_camera=False)

        elif self.pick_mode == 'plane' and len(pts) >= 2:
            # Draw lines connecting picked points so far
            line_pts = []
            for i in range(len(pts) - 1):
                line_pts += [2, i, i + 1]
            if len(pts) == 3:
                line_pts += [2, 2, 0]  # close triangle
            poly = pv.PolyData(np.array(pts))
            poly.lines = np.array(line_pts)
            self.plotter.add_mesh(poly, name='_pick_lines', color='cyan', line_width=3, reset_camera=False)

            if len(pts) == 3:
                # Draw normal arrow
                p1, p2, p3 = pts
                n = np.cross(p2 - p1, p3 - p1)
                n_len = np.linalg.norm(n)
                if n_len > 1e-9:
                    n /= n_len
                    if n[1] < 0: n = -n
                    centre = (p1 + p2 + p3) / 3.0
                    arrow_len = np.linalg.norm(p2 - p1) * 0.6
                    arrow = pv.Arrow(start=centre, direction=n, scale=arrow_len)
                    self.plotter.add_mesh(arrow, name='_pick_normal', color='lime', reset_camera=False)

        self.plotter.render()

    def pick_callback(self, pt):
        self.picked_pts.append(np.array(pt))
        self._draw_pick_visuals()

        # Modes that act immediately on first pick
        if self.pick_mode == 'xyz_origin':
            for i, k in enumerate(['TransX', 'TransY', 'TransZ']):
                self.srt_ctrl[k].setValue(self.srt_ctrl[k].value() - pt[i])
            self.finish_pick(); return
        elif self.pick_mode == 'xz_origin':
            self.srt_ctrl['TransX'].setValue(self.srt_ctrl['TransX'].value() - pt[0])
            self.srt_ctrl['TransZ'].setValue(self.srt_ctrl['TransZ'].value() - pt[2])
            self.finish_pick(); return

        # Multi-point modes: wait for confirm once enough points collected
        elif self.pick_mode == 'align' and len(self.picked_pts) == 2:
            self.plotter.disable_picking()
            self.plotter.add_text("✅ Left-click viewport to confirm  |  Right-click to cancel",
                                  name='_pick_msg', position='lower_edge', font_size=10, color='yellow')
            self.plotter.iren.add_observer("LeftButtonPressEvent",  self._confirm_pick)
            self.plotter.iren.add_observer("RightButtonPressEvent", self._cancel_pick)

        elif self.pick_mode == 'plane' and len(self.picked_pts) == 3:
            self.plotter.disable_picking()
            self.plotter.add_text("✅ Left-click viewport to confirm  |  Right-click to cancel",
                                  name='_pick_msg', position='lower_edge', font_size=10, color='yellow')
            self.plotter.iren.add_observer("LeftButtonPressEvent",  self._confirm_pick)
            self.plotter.iren.add_observer("RightButtonPressEvent", self._cancel_pick)

    def _confirm_pick(self, obj, event):
        self.plotter.iren.remove_observers("LeftButtonPressEvent")
        self.plotter.iren.remove_observers("RightButtonPressEvent")
        self.plotter.remove_actor('_pick_msg')
        pts = self.picked_pts

        if self.pick_mode == 'align' and len(pts) == 2:
            p1, p2 = pts
            ang = np.degrees(np.arctan2(p2[0] - p1[0], p2[2] - p1[2]))
            self.srt_ctrl['RotY'].setValue(self.srt_ctrl['RotY'].value() - ang)

        elif self.pick_mode == 'plane' and len(pts) == 3:
            p1, p2, p3 = pts
            n = np.cross(p2 - p1, p3 - p1)
            n_len = np.linalg.norm(n)
            if n_len < 1e-9: self.finish_pick(); return
            n /= n_len
            if n[1] < 0: n = -n
            target = np.array([0.0, 1.0, 0.0])
            axis = np.cross(n, target)
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-9: self.finish_pick(); return
            axis /= axis_len
            angle = np.arccos(np.clip(np.dot(n, target), -1, 1))
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
            ry = np.degrees(np.arctan2( R[0, 2], np.sqrt(R[1,2]**2 + R[2,2]**2)))
            rx = np.degrees(np.arctan2(-R[1, 2], R[2, 2]))
            rz = np.degrees(np.arctan2(-R[0, 1], R[0, 0]))
            self.srt_ctrl['RotX'].setValue(self.srt_ctrl['RotX'].value() + rx)
            self.srt_ctrl['RotY'].setValue(self.srt_ctrl['RotY'].value() + ry)
            self.srt_ctrl['RotZ'].setValue(self.srt_ctrl['RotZ'].value() + rz)

        self.finish_pick()

    def _cancel_pick(self, obj, event):
        self.plotter.iren.remove_observers("LeftButtonPressEvent")
        self.plotter.iren.remove_observers("RightButtonPressEvent")
        self.plotter.remove_actor('_pick_msg')
        self.finish_pick()

    def finish_pick(self):
        self.plotter.disable_picking()
        self._clear_pick_visuals()
        self.pick_mode = None
        for b in self.pick_btn_map.values(): b.setChecked(False)
        self.on_srt_change()

    def open_file(self):
        f = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not f: return
        self.proj = COLMAPProject(); self.proj.load(f)
        self._culled_ids = set()
        if not self.proj.points3D:
            QMessageBox.warning(self, "Empty Project",
                "No 3D points found in the selected folder.\n"
                "Make sure it contains points3D.bin or points3D.txt.")
            return
        pts = np.vstack([v['xyz'] * [-1, -1, 1] for v in self.proj.points3D.values()]).astype(np.float32)
        rgbs = np.vstack([v['rgb'] for v in self.proj.points3D.values()]).astype(np.uint8)
        self.cloud_poly = pv.PolyData(pts); self.cloud_poly.point_data["rgb"] = rgbs
        self.radio_bin.setChecked(self.proj.is_bin)
        self.radio_txt.setChecked(not self.proj.is_bin)
        self.bounds = [float(pts[:,0].min()), float(pts[:,0].max()), float(pts[:,1].min()), float(pts[:,1].max()), float(pts[:,2].min()), float(pts[:,2].max())]
        self.reset_crop()

    def toggle_origin(self, show):
        self.plotter.remove_actor("origin")
        if show and self.cloud_poly:
            b = np.array(self.cloud_poly.bounds); al = (b[1]-b[0]) * 0.3
            pts = np.array([[0,0,0], [-al,0,0], [0,al,0], [0,0,-al]], float)
            o = pv.PolyData(pts, lines=np.array([2,0,1, 2,0,2, 2,0,3]))
            o.cell_data["colors"] = np.array([[255,0,0],[0,255,0],[0,0,255]], dtype=np.uint8)
            self.plotter.add_mesh(o, name="origin", scalars="colors", rgb=True, line_width=5)

    def open_in_npp(self): subprocess.Popen([r"C:\Program Files\Notepad++\notepad++.exe", self.settings_file]) if os.path.exists(r"C:\Program Files\Notepad++\notepad++.exe") else os.startfile(self.settings_file)
    def bake_srt(self):
        if not self.cloud_poly: return
        # Apply transform to display mesh
        self.cloud_poly = self.cloud_poly.transform(self.get_mat(), inplace=True)
        # Also update the stored COLMAP-space xyz values so export stays in sync
        S, R_c, T_c = self.get_colmap_srt()
        for p in self.proj.points3D.values():
            p['xyz'] = S * (R_c @ p['xyz']) + T_c
        # Also bake camera poses
        for img in self.proj.images.values():
            R_cw = qvec2rotmat(img['qvec'])
            t_cw = np.array(img['tvec'])
            C        = -R_cw.T @ t_cw
            C_new    = S * (R_c @ C) + T_c
            R_cw_new = R_cw @ R_c.T
            t_cw_new = -R_cw_new @ C_new
            img['qvec'] = rotmat2qvec(R_cw_new)
            img['tvec'] = t_cw_new
        for k, s in self.srt_ctrl.items():
            s.blockSignals(True); s.setValue(1.0 if k == 'Scale' else 0.0); s.blockSignals(False)
        self.set_bake_state(False); self.update_preview()
        self._write_logfile(event="bake")
    def _write_logfile(self, event="bake", output_folder=None, export_fmt=None,
                       points_exported=None, points_total=None):
        """Write logfile.json next to PyVista_Colmap_settings.json recording what was applied."""
        import datetime
        log_path = os.path.join(os.path.dirname(self.settings_file), "logfile.json")

        # Load existing log (list of entries) or start fresh
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f: log = json.load(f)
                if not isinstance(log, list): log = []
            except Exception: log = []
        else:
            log = []

        # Capture current SRT values (pre-reset for bake, live values for export)
        srt = {k: float(v.value()) for k, v in self.srt_ctrl.items()}

        # Capture current crop state
        axis_labels = ["X", "Y", "Z"]
        crop = {}
        for i, ax in enumerate(axis_labels):
            crop[ax] = {"min": float(self.current_crop[2*i]),
                        "max": float(self.current_crop[2*i+1])}

        entry = {
            "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
            "event":         event,
            "source_file":   str(self.proj.load.__self__.__class__.__name__) if self.proj else None,
            "points_total":  len(self.proj.points3D) if self.proj else 0,
            "cameras_total": len(self.proj.cameras)  if self.proj else 0,
            "images_total":  len(self.proj.images)   if self.proj else 0,
            "fov":           float(self.spin_fov.value()),
            "srt": {
                "Scale":  srt.get("Scale",  1.0),
                "RotX":   srt.get("RotX",   0.0),
                "RotY":   srt.get("RotY",   0.0),
                "RotZ":   srt.get("RotZ",   0.0),
                "TransX": srt.get("TransX", 0.0),
                "TransY": srt.get("TransY", 0.0),
                "TransZ": srt.get("TransZ", 0.0),
            },
            "crop": crop,
        }
        if output_folder:   entry["output_folder"]    = output_folder
        if export_fmt:      entry["export_fmt"]        = export_fmt
        if points_total is not None:
            entry["points_total"]    = points_total
            entry["points_exported"] = points_exported if points_exported is not None else points_total
            entry["points_cropped"]  = points_total - (points_exported or points_total)

        log.append(entry)
        try:
            with open(log_path, "w") as f: json.dump(log, f, indent=2)
            print(f"[log] Written → {log_path}")
        except Exception as e:
            print(f"[log] Failed to write logfile: {e}")

    def on_srt_change(self):
        self.btn_bake.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold;")
        self.btn_bake.setText("⚠️ BAKE REQUIRED")
        if not hasattr(self, '_srt_timer'):
            from PySide6.QtCore import QTimer
            self._srt_timer = QTimer(); self._srt_timer.setSingleShot(True)
            self._srt_timer.timeout.connect(self.update_preview)
        self._srt_timer.start(50)  # ms debounce — fires once after slider stops
    def set_bake_state(self, d):
        self.btn_bake.setStyleSheet(f"background-color: {'#d32f2f' if d else '#2e7d32'}; color: white; font-weight: bold;")
        self.btn_bake.setText("⚠️ BAKE REQUIRED" if d else "✅ SRT Baked")
    def update_step_sizes(self): [c.setSingleStep(self.spin_step.value()) for c in self.srt_ctrl.values()]
    def sync_crop_ui(self):
        idx = self.axis_sel.currentIndex(); d_min, d_max = self.bounds[2*idx], self.bounds[2*idx+1]
        for l in ['min', 'max']:
            self.crop_ui[l].blockSignals(True); self.crop_ui[l].setRange(d_min-5000, d_max+5000)
            self.crop_ui[l].setValue(self.current_crop[2*idx+(0 if l=='min' else 1)]); self.crop_ui[l].blockSignals(False)
        self.update_preview()
    def on_slider_move(self):
        idx = self.axis_sel.currentIndex(); d_min, d_max = self.bounds[2*idx], self.bounds[2*idx+1]
        for l in ['min', 'max']: self.crop_ui[l].setValue(d_min + (self.crop_ui[f"{l}_sl"].value()/1000.0)*(d_max-d_min))
    def auto_crop_stat(self, pct):
        for i in range(3): d = self.cloud_poly.points[:,i]; self.current_crop[2*i], self.current_crop[2*i+1] = float(np.percentile(d, (1-pct)/2*100)), float(np.percentile(d, (1-(1-pct)/2)*100))
        self.sync_crop_ui()
    def reset_crop(self):
        if not self.cloud_poly: return
        transformed = self.cloud_poly.copy().transform(self.get_mat(), inplace=True)
        pts = transformed.points
        self.bounds = [float(pts[:,0].min()), float(pts[:,0].max()),
                       float(pts[:,1].min()), float(pts[:,1].max()),
                       float(pts[:,2].min()), float(pts[:,2].max())]
        self.current_crop = list(self.bounds)
        self.sync_crop_ui()
    def apply_view_preset(self, idx):
        # position, focal_point, view_up
        presets = [
            ((-200,200,-200),  (0,0,0), (0,1,0)),   # 7 NW
            ((0,0, -200),  (0,0,0), (0,1,0)),   # 8 BACK
            ((200,200,-200),  (0,0,0), (0,1,0)),   # 9 NE
            ((-200,0,0),  (0,0,0), (0,1,0)),   # 4 LEFT
            ((0, 200,0),  (0,0,0), (0,0,-1)),  # 5 TOP  (180° rotated — up = -Z)
            ((200,0,0),(0,0,0),(0,1,0)), # 6 RIGHT
            ((-200,200, 200),(0,0,0),(0,1,0)), # 1 SW
            ((0,0,200),(0,0,0),(0,1,0)), # 2 FRONT
            (( 200,200, 200),(0,0,0),(0,1,0)), # 3 SE
        ]
        pos, foc, up = presets[idx]
        self.plotter.camera_position = [pos, foc, up]; self.plotter.reset_camera()

    def closeEvent(self, event):
        for p in (self.plotter, self.hist_plotter):
            try:
                rw = p.render_window
                if rw is not None:
                    rw.Finalize()
            except Exception:
                pass
            try:
                p.close()
            except Exception:
                pass
        # Allow a fresh launch after a proper close
        app = QApplication.instance()
        if app is not None and getattr(app, '_colmap_window', None) is self:
            app._colmap_window = None
        event.accept()

def _run():
    app = QApplication.instance() or _app

    if _session_already_launched():
        _notify_already_running()
        return

    splash = _splash   # already showing since module load

    ex = COLMAPExplorer()

    if splash is not None:
        splash.finish(ex)
    ex.show()

    if not hasattr(app, '_colmap_running'):
        app._colmap_running = True
        sys.exit(app.exec())

if __name__ == "__main__":
    _run()