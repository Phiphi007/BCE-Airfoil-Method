import numpy as np
import math
import os
import glob
import subprocess
import time
import uuid
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import matplotlib.gridspec as gridspec

# ==========================================================================
# 1.XFOIL 多迎角极曲线求解总线
# ==========================================================================
def run_xfoil(x_coords, y_coords, alphas=np.linspace(-5, 10, 16), re=500000, mach=0.0):
    uid = uuid.uuid4().hex[:8]
    dat_file = f"temp_{uid}.dat"
    polar_file = f"polar_{uid}.txt"
    in_file = f"xfoil_{uid}.in"

    x_up = x_coords[::-1]
    y_up = y_coords[0][::-1]
    x_dn = x_coords[1:]
    y_dn = y_coords[1][1:]
    
    with open(dat_file, "w") as f:
        f.write(f"BCE_Station_Airfoil_{uid}\n")
        for x, y in zip(x_up, y_up): f.write(f"{x:.6f} {y:.6f}\n")
        for x, y in zip(x_dn, y_dn): f.write(f"{x:.6f} {y:.6f}\n")

    xfoil_script = f"""
LOAD {dat_file}
PANE
OPER
Visc {re}
Mach {mach}
ITER 200
PACC
{polar_file}

Aseq {alphas[0]} {alphas[-1]} {alphas[1]-alphas[0]}
Pxx
QUIT
"""
    with open(in_file, "w") as f: f.write(xfoil_script)

    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = subprocess.CREATE_NO_WINDOW
    else:
        startupinfo = None
        creationflags = 0

    try:
        subprocess.run(f"xfoil.exe < {in_file}", shell=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                       timeout=4.0, startupinfo=startupinfo, creationflags=creationflags)
    except subprocess.TimeoutExpired:
        pass

    res = None
    if os.path.exists(polar_file):
        try:
            data = np.loadtxt(polar_file, skiprows=12)
            if len(data) > 0:
                if data.ndim == 1: data = data.reshape(1, -1)
                res = (data[:, 0], data[:, 1], data[:, 2], data[:, 1] / np.maximum(data[:, 2], 1e-5))
        except:
            res = None

    for fl in [dat_file, polar_file, in_file]:
        if os.path.exists(fl):
            try: os.remove(fl)
            except: pass
    return res

# ==========================================================================
# 2. 21维 原生 BCE 核心数学与物理解耦引擎
# ==========================================================================
def bezier_2d(CPx, CPy, num_points=2000):
    n = len(CPx) - 1
    t = np.linspace(0, 1, num_points)
    x, y = np.zeros_like(t), np.zeros_like(t)
    for i in range(n + 1):
        bernstein = math.comb(n, i) * (t**i) * ((1 - t)**(n - i))
        x += CPx[i] * bernstein
        y += CPy[i] * bernstein
    return x, y

def get_bce_functions(p):
    CPy_h = list(p[0:7]) + [0.0]; CPy_a = list(p[7:14]); CPy_b = list(p[14:20])
    gamma = p[20]
    CPx_h = [(i / 7)**gamma for i in range(8)]
    CPx_a = [(i / 6)**gamma for i in range(7)]
    CPx_b = [(i / 5)**gamma for i in range(6)]
    return bezier_2d(CPx_h, CPy_h), bezier_2d(CPx_a, CPy_a), bezier_2d(CPx_b, CPy_b), {'h': (CPx_h, CPy_h), 'a': (CPx_a, CPy_a), 'b': (CPx_b, CPy_b)}

def get_bce_airfoil(p, X_eval_targets, mode='Free', Zt_ref=None, Yc_ref=None):
    h_curve, a_curve, b_curve, _ = get_bce_functions(p)
    t_eval = np.linspace(0, 1, 2000)
    h_val = np.interp(t_eval, h_curve[0], h_curve[1]) 
    alpha_val = np.radians(np.interp(t_eval, a_curve[0], a_curve[1]))
    beta_val = np.radians(np.maximum(np.interp(t_eval, b_curve[0], b_curve[1]), 0.1))
    
    denom = np.maximum(np.cos(beta_val)**2 - np.sin(alpha_val)**2, 1e-4)
    yc_raw = h_val * (np.sin(alpha_val) * np.cos(alpha_val)) / denom
    ay_raw = h_val * np.sin(beta_val) * np.cos(beta_val) / denom
    ax_raw = h_val * np.sin(beta_val) / np.sqrt(denom)

    if mode == 'Lock Zt' and Zt_ref is not None:
        ay_raw = Zt_ref
        ax_raw = ay_raw / np.maximum(np.cos(beta_val), 1e-4) 
    elif mode == 'Lock Yc' and Yc_ref is not None:
        yc_raw = Yc_ref

    ax_raw_abs = np.maximum(np.abs(ax_raw), 1e-6)
    left_bounds, right_bounds = t_eval - ax_raw_abs, t_eval + ax_raw_abs
    idx_LE, idx_TE = np.argmin(left_bounds), np.argmax(right_bounds)
    
    chord = np.maximum(np.sqrt((right_bounds[idx_TE] - left_bounds[idx_LE])**2 + (yc_raw[idx_TE] - yc_raw[idx_LE])**2), 1e-8)
    theta = -np.arctan2(yc_raw[idx_TE] - yc_raw[idx_LE], right_bounds[idx_TE] - left_bounds[idx_LE])

    X_raw_eval = left_bounds[idx_LE] + (right_bounds[idx_TE] - left_bounds[idx_LE]) * t_eval
    Y_up_raw, Y_dn_raw = np.zeros_like(X_raw_eval), np.zeros_like(X_raw_eval)
    
    for i, X in enumerate(X_raw_eval):
        mask = np.abs(X - t_eval) <= ax_raw_abs
        if np.any(mask):
            term = np.sqrt(np.maximum(0, 1.0 - ((X - t_eval[mask]) / ax_raw_abs[mask])**2))
            Y_up_raw[i] = np.max(yc_raw[mask] + ay_raw[mask] * term)
            Y_dn_raw[i] = np.min(yc_raw[mask] - ay_raw[mask] * term)
        else: Y_up_raw[i] = Y_dn_raw[i] = yc_raw[i]

    def transform(x, y):
        xs, ys = x - left_bounds[idx_LE], y - yc_raw[idx_LE]
        return (xs * np.cos(theta) - ys * np.sin(theta)) / chord, (xs * np.sin(theta) + ys * np.cos(theta)) / chord

    X_up_norm, Y_up_norm = transform(X_raw_eval, Y_up_raw)
    X_dn_norm, Y_dn_norm = transform(X_raw_eval, Y_dn_raw)
    X_cam_norm, Y_cam_norm = transform(t_eval, yc_raw)

    return np.interp(X_eval_targets, X_up_norm[np.argsort(X_up_norm)], Y_up_norm[np.argsort(X_up_norm)]), \
           np.interp(X_eval_targets, X_dn_norm[np.argsort(X_dn_norm)], Y_dn_norm[np.argsort(X_dn_norm)]), \
           np.interp(X_eval_targets, X_cam_norm[np.argsort(X_cam_norm)], Y_cam_norm[np.argsort(X_cam_norm)])

# ==========================================
# 3. 后缘切线延长修尖引擎
# ==========================================
def get_naca2412():
    X_t = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, 160)))
    yt = 5 * 0.12 * (0.2969*np.sqrt(X_t) - 0.1260*X_t - 0.3516*X_t**2 + 0.2843*X_t**3 - 0.1015*X_t**4)
    yc = np.where(X_t < 0.4, 0.02 / (0.4**2) * (2 * 0.4 * X_t - X_t**2), 0.02 / (0.6**2) * (1 - 2 * 0.4 + 2 * 0.4 * X_t - X_t**2))
    return X_t, yc + yt, yc - yt

def process_dat_file_sharpen(filename, num_eval=160):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try: coords.append([float(parts[0]), float(parts[1])])
                except ValueError: continue
    coords = np.array(coords)
    idx_LE = np.argmin(coords[:, 0])
    upper_raw = coords[:idx_LE+1][::-1]
    lower_raw = coords[idx_LE:]
    upper_raw = upper_raw[np.unique(upper_raw[:,0], return_index=True)[1]]
    lower_raw = lower_raw[np.unique(lower_raw[:,0], return_index=True)[1]]

    TE_up, TE_dn = upper_raw[-1], lower_raw[-1]
    te_thickness = math.sqrt((TE_up[0]-TE_dn[0])**2 + (TE_up[1]-TE_dn[1])**2)
    
    if te_thickness > 1e-4:
        print(f"[*] 发现钝后缘 (厚度: {te_thickness:.5f})，正在执行切线闭合修尖...")
        p_u1, p_u0 = upper_raw[-2], upper_raw[-1]
        p_l1, p_l0 = lower_raw[-2], lower_raw[-1]
        m_u = (p_u0[1] - p_u1[1]) / (p_u0[0] - p_u1[0] + 1e-8)
        m_l = (p_l0[1] - p_l1[1]) / (p_l0[0] - p_l1[0] + 1e-8)
        if abs(m_u - m_l) > 1e-5:
            X_int = (m_u * p_u0[0] - m_l * p_l0[0] - p_u0[1] + p_l0[1]) / (m_u - m_l)
            Y_int = m_u * (X_int - p_u0[0]) + p_u0[1]
            virtual_TE = np.array([X_int, Y_int])
        else:
            virtual_TE = np.array([p_u0[0] + 0.005, (p_u0[1]+p_l0[1])/2])
        upper_raw = np.vstack([upper_raw, virtual_TE])
        lower_raw = np.vstack([lower_raw, virtual_TE])

    LE = upper_raw[0]
    chord = np.maximum(np.sqrt((upper_raw[-1, 0] - LE[0])**2 + (upper_raw[-1, 1] - LE[1])**2), 1e-8)
    theta = -np.arctan2(upper_raw[-1, 1] - LE[1], upper_raw[-1, 0] - LE[0])
    
    def transform(pts):
        return ((pts[:, 0] - LE[0]) * np.cos(theta) - (pts[:, 1] - LE[1]) * np.sin(theta)) / chord, \
               ((pts[:, 0] - LE[0]) * np.sin(theta) + (pts[:, 1] - LE[1]) * np.cos(theta)) / chord
    
    ux, uy = transform(upper_raw); lx, ly = transform(lower_raw)
    X_target = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_eval)))
    Y_up = interp1d(ux, uy, kind='cubic', fill_value='extrapolate')(X_target)
    Y_dn = interp1d(lx, ly, kind='cubic', fill_value='extrapolate')(X_target)
    Y_up[0] = Y_dn[0] = Y_up[-1] = Y_dn[-1] = 0.0
    return X_target, Y_up, Y_dn

# ==========================================
# 4. 工作目录扫描与拟合器
# ==========================================
print("="*70)
print("  🚀 BCE 气动一体化工作站（大尺度交互控制版）")
print("="*70)

dat_files = glob.glob("*.dat")
if "temp_airfoil.dat" in dat_files: dat_files.remove("temp_airfoil.dat")

if len(dat_files) == 0:
    print("[*] 未在当前工作目录发现外部 .dat 文件，默认载入 NACA 2412。")
    X_target, Y_up_target, Y_dn_target = get_naca2412()
    selected_title = "NACA 2412 Baseline"
else:
    print("\n[+] 成功扫描当前工作目录，发现以下翼型文件：")
    for idx, f_name in enumerate(dat_files):
        print(f"   [{idx}] {f_name}")
    try:
        user_sel = input(f"请输入序列号选择翼型 (0-{len(dat_files)-1}, 直接回车默认NACA2412): ").strip()
        if user_sel == "":
            X_target, Y_up_target, Y_dn_target = get_naca2412()
            selected_title = "NACA 2412 Baseline"
        else:
            sel_idx = int(user_sel)
            X_target, Y_up_target, Y_dn_target = process_dat_file_sharpen(dat_files[sel_idx])
            selected_title = f"Target: {dat_files[sel_idx]}"
    except:
        X_target, Y_up_target, Y_dn_target = get_naca2412()
        selected_title = "NACA 2412 Baseline"

def residual_function(p):
    try:
        Y_up_bce, Y_dn_bce, _ = get_bce_airfoil(p, X_target)
        weight = np.ones_like(X_target)
        weight[:25] = 4.0 
        return np.concatenate([(Y_up_bce - Y_up_target)*weight, (Y_dn_bce - Y_dn_target)*weight])
    except:
        return np.ones(len(X_target)*2) * 1e3

p_init = [0.05, 0.10, 0.12, 0.15, 0.12, 0.08, 0.02] + [0.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0] + [25.0, 20.0, 15.0, 10.0, 8.0, 5.0] + [1.0]
bounds = ([-0.25]*7 + [-45.0]*7 + [2.0]*6 + [0.3], [0.45]*7 + [45.0]*7 + [70.0]*6 + [3.0])
res = least_squares(residual_function, p_init, bounds=bounds, method='trf', ftol=1e-8)
p_opt = res.x

# ==========================================
# 5. 全窗口超大视场自适应交互画板
# ==========================================
class AdvancedBCEStation:
    def __init__(self, p_base):
        self.p_baseline = p_base.copy()
        self.p = self.p_baseline.copy()
        
        self.fig = plt.figure(figsize=(17, 9.5))
        self.fig.canvas.manager.set_window_title('BCE Aerodynamic Integration Platform')
        
        gs = gridspec.GridSpec(2, 4, figure=self.fig, width_ratios=[0.8, 1.1, 1.1, 1.1], wspace=0.34, hspace=0.40)
        
        self.ax_geom = self.fig.add_subplot(gs[0, 1])
        self.ax_h = self.fig.add_subplot(gs[1, 1])
        self.ax_a = self.fig.add_subplot(gs[0, 2])
        self.ax_b = self.fig.add_subplot(gs[1, 2])
        self.ax_cl = self.fig.add_subplot(gs[0, 3])
        self.ax_ld = self.fig.add_subplot(gs[1, 3])
        
        self.mode = 'Free'
        self.Zt_ref, self.Yc_ref = None, None
        self.compute_reference_fields()
        
        self.fig.patches.extend([plt.Rectangle((0.01, 0.04), 0.165, 0.88, fill=True, color='#f1f2f6', alpha=0.9, transform=self.fig.transFigure, zorder=-1)])
        
        ax_lbl = plt.axes([0.02, 0.82, 0.15, 0.05]); ax_lbl.axis('off')
        ax_lbl.text(0, 0.5, "Geometric Constraints", fontsize=11, fontweight='bold', color='#2f3542')
        
        ax_radio = plt.axes([0.02, 0.68, 0.14, 0.13], facecolor='none')
        self.radio = RadioButtons(ax_radio, ('Free Mode', 'Lock Zt (Thickness)', 'Lock Yc (Camber)'))
        self.radio.on_clicked(self.on_mode_change)
            
        ax_reset = plt.axes([0.02, 0.52, 0.14, 0.045])
        self.btn_reset = Button(ax_reset, 'Reset Geometry', color='#ff4757', hovercolor='#ff6b6b')
        self.btn_reset.label.set_fontsize(10); self.btn_reset.label.set_color('white'); self.btn_reset.label.set_fontweight('bold')
        self.btn_reset.on_clicked(self.on_reset)

        ax_compute = plt.axes([0.02, 0.42, 0.14, 0.055])
        self.btn_compute = Button(ax_compute, 'Compute XFoil', color='#2ed573', hovercolor='#26af5f')
        self.btn_compute.label.set_fontsize(11); self.btn_compute.label.set_color('white'); self.btn_compute.label.set_fontweight('bold')
        self.btn_compute.on_clicked(self.on_compute_clicked)

        self.init_plots()
        self.update_title_status()

    def compute_reference_fields(self):
        h_curve, a_curve, b_curve, _ = get_bce_functions(self.p)
        t_eval = np.linspace(0, 1, 2000)
        h_val = np.interp(t_eval, h_curve[0], h_curve[1])
        alpha_val = np.radians(np.interp(t_eval, a_curve[0], a_curve[1]))
        beta_val = np.radians(np.maximum(np.interp(t_eval, b_curve[0], b_curve[1]), 0.1))
        denom = np.maximum(1e-4, np.cos(beta_val)**2 - np.sin(alpha_val)**2)
        self.Zt_ref = h_val * np.sin(beta_val) * np.cos(beta_val) / denom
        self.Yc_ref = h_val * (np.sin(alpha_val) * np.cos(alpha_val)) / denom

    def update_title_status(self):
        self.fig.suptitle(f"BCE Aerodynamic Optimization Platform ({selected_title})\nCurrent State: [Mode = {self.mode}]  •  Modify handles dynamically and press Compute", 
                          fontsize=13, fontweight='bold', color='#1e272e', y=0.98)
        self.fig.canvas.draw_idle()

    def init_plots(self):
        self.ax_cl.set_title("Lift Coefficient ($C_L/alpha$)", fontsize=11, fontweight='bold', pad=10)
        self.ax_cl.grid(True, linestyle=':', alpha=0.5); self.ax_cl.set_xlim(-5, 10); self.ax_cl.set_ylim(-0.4, 1.6)
        self.line_cl, = self.ax_cl.plot([], [], '#2ecc71', linewidth=2.5, label='Live Airfoil')
        self.line_cl_base, = self.ax_cl.plot([], [], 'k--', linewidth=1.2, alpha=0.4, label='Baseline')
        self.ax_cl.legend(loc='lower right', fontsize=8)
        
        self.ax_ld.set_title("Lift-to-Drag Efficiency ($L/D$)", fontsize=11, fontweight='bold', pad=10)
        self.ax_ld.grid(True, linestyle=':', alpha=0.5); self.ax_ld.set_xlim(-5, 10); self.ax_ld.set_ylim(-15, 130)
        self.line_ld, = self.ax_ld.plot([], [], '#e67e22', linewidth=2.5, label='Live Airfoil')
        self.line_ld_base, = self.ax_ld.plot([], [], 'k--', linewidth=1.2, alpha=0.4, label='Baseline')
        self.ax_ld.legend(loc='upper right', fontsize=8)

        self.ax_geom.set_title("Airfoil Profile Contour", fontsize=11, fontweight='bold', pad=10)
        self.ax_geom.plot(X_target, Y_up_target, color='#bdc3c7', linestyle='--', linewidth=2)
        self.ax_geom.plot(X_target, Y_dn_target, color='#bdc3c7', linestyle='--', linewidth=2)
        self.ax_geom.plot([0,1], [0,0], 'k-', linewidth=0.5)
        
        Y_up, Y_dn, Y_cam = get_bce_airfoil(self.p, X_target, mode=self.mode, Zt_ref=self.Zt_ref, Yc_ref=self.Yc_ref)
        self.line_up, = self.ax_geom.plot(X_target, Y_up, '#2980b9', linewidth=2.5)
        self.line_dn, = self.ax_geom.plot(X_target, Y_dn, '#27ae60', linewidth=2.5)
        self.line_cam, = self.ax_geom.plot(X_target, Y_cam, '#f1c40f', linestyle='--', linewidth=1.5)
        self.ax_geom.axis('equal'); self.ax_geom.grid(True, linestyle=':', alpha=0.5)

        h_curve, a_curve, b_curve, ctrls = get_bce_functions(self.p)
        
        self.ax_h.set_title("Thickness Form Function $h(x)$", fontsize=10, pad=8)
        self.line_h, = self.ax_h.plot(h_curve[0], h_curve[1], 'k-', linewidth=2)
        self.poly_h, = self.ax_h.plot(ctrls['h'][0], ctrls['h'][1], '#e74c3c', linewidth=1, alpha=0.4) 
        self.pts_h, = self.ax_h.plot(ctrls['h'][0][:-1], ctrls['h'][1][:-1], 'ro', markersize=8, picker=True, pickradius=10)
        self.ax_h.grid(True, linestyle=':', alpha=0.5)

        self.ax_a.set_title("Skeletal Camber Distribution $\\alpha(x)$", fontsize=10, pad=8)
        self.line_a, = self.ax_a.plot(a_curve[0], a_curve[1], '#9b59b6', linewidth=2)
        self.poly_a, = self.ax_a.plot(ctrls['a'][0], ctrls['a'][1], '#e74c3c', linewidth=1, alpha=0.4)
        self.pts_a, = self.ax_a.plot(ctrls['a'][0], ctrls['a'][1], 'ro', markersize=8, picker=True, pickradius=10)
        self.ax_a.axhline(0, color='k', linestyle='--', linewidth=0.8)
        self.ax_a.grid(True, linestyle=':', alpha=0.5)

        self.ax_b.set_title("Leading-edge Fullness Function $\\beta(x)$", fontsize=10, pad=8)
        self.line_b, = self.ax_b.plot(b_curve[0], b_curve[1], '#f39c12', linewidth=2)
        self.poly_b, = self.ax_b.plot(ctrls['b'][0], ctrls['b'][1], '#e74c3c', linewidth=1, alpha=0.4)
        self.pts_b, = self.ax_b.plot(ctrls['b'][0], ctrls['b'][1], 'ro', markersize=8, picker=True, pickradius=10)
        self.ax_b.grid(True, linestyle=':', alpha=0.5)

        self.active_var_idx, self.active_artist = None, None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.optimize_axes_limits()
        self.execute_xfoil_calculation(is_initial=True)

    def optimize_axes_limits(self):
        """ 
        【超大基础视场 + 自适应包络】：
        提供远大于原本包络的基础画幅边界，彻底解放大尺度正负拖拽！
        """
        h_pts = self.p[0:7]
        a_pts = self.p[7:14]
        b_pts = self.p[14:20]

        # h 轴：基础视场设为 [-0.15, 0.50]，如果被突破再向外扩大
        h_min, h_max = min(h_pts), max(h_pts)
        self.ax_h.set_ylim(min(-0.15, h_min - 0.05), max(0.50, h_max + 0.05))

        # α 轴：基础视场扩大至 [-35, 35] 极其宽阔的弯度变形空间
        a_min, a_max = min(a_pts), max(a_pts)
        limit_a = max(35.0, abs(a_min)*1.2, abs(a_max)*1.2)
        self.ax_a.set_ylim(-limit_a, limit_a)

        # β 轴：基础视场扩大至 [0, 75]
        b_min, b_max = min(b_pts), max(b_pts)
        self.ax_b.set_ylim(0, max(75.0, b_max + 5.0))

    def on_mode_change(self, label):
        if 'Free' in label:
            self.mode = 'Free'
        elif 'Lock Zt' in label:
            self.mode = 'Lock Zt'
        elif 'Lock Yc' in label:
            self.mode = 'Lock Yc'
            
        self.compute_reference_fields()
        self.update_title_status()
        self.update_plots()
        
    def on_reset(self, event):
        self.p = self.p_baseline.copy()
        self.compute_reference_fields()
        self.update_plots()

    def on_compute_clicked(self, event):
        self.execute_xfoil_calculation(is_initial=False)

    def execute_xfoil_calculation(self, is_initial=False):
        Y_up, Y_dn, _ = get_bce_airfoil(self.p, X_target, mode=self.mode, Zt_ref=self.Zt_ref, Yc_ref=self.Yc_ref)
        polar = run_xfoil(X_target, (Y_up, Y_dn))
        if polar is not None:
            a_res, cl_res, _, ld_res = polar
            if is_initial:
                self.line_cl_base.set_data(a_res, cl_res)
                self.line_ld_base.set_data(a_res, ld_res)
            self.line_cl.set_data(a_res, cl_res)
            self.line_ld.set_data(a_res, ld_res)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes is None: return
        for ax, pts_artist, offset in zip([self.ax_h, self.ax_a, self.ax_b], [self.pts_h, self.pts_a, self.pts_b], [0, 7, 14]):
            if event.inaxes == ax:
                cont, ind = pts_artist.contains(event)
                if cont:
                    point_idx = offset + ind["ind"][0]
                    if self.mode == 'Lock Zt' and point_idx < 7:
                        print("  -> ⚠️ 几何锁死激活：Thickness分布由系统接管！")
                        return
                    if self.mode == 'Lock Yc' and 7 <= point_idx < 14:
                        print("  -> ⚠️ 几何锁死激活：Camber骨架由系统接管！")
                        return
                    self.active_var_idx, self.active_artist = point_idx, pts_artist
                    break

    def on_motion(self, event):
        if self.active_var_idx is None or event.inaxes is None: return
        new_y = event.ydata
        
        # 解除对 h(x) 和 α(x) 的硬性防越界裁剪，交由底层的极大边界管控
        if self.active_var_idx >= 14: 
            new_y = max(2.0, min(80.0, new_y))
            
        self.p[self.active_var_idx] = new_y
        self.update_plots()

    def update_plots(self):
        try: Y_up, Y_dn, Y_cam = get_bce_airfoil(self.p, X_target, mode=self.mode, Zt_ref=self.Zt_ref, Yc_ref=self.Yc_ref)
        except: return 
        
        self.line_up.set_ydata(Y_up); self.line_dn.set_ydata(Y_dn); self.line_cam.set_ydata(Y_cam)
        h_curve, a_curve, b_curve, ctrls = get_bce_functions(self.p)
        
        self.line_h.set_ydata(h_curve[1]); self.pts_h.set_ydata(ctrls['h'][1][:-1]); self.poly_h.set_ydata(ctrls['h'][1])
        self.line_a.set_ydata(a_curve[1]); self.pts_a.set_ydata(ctrls['a'][1]); self.poly_a.set_ydata(ctrls['a'][1])
        self.line_b.set_ydata(b_curve[1]); self.pts_b.set_ydata(ctrls['b'][1]); self.poly_b.set_ydata(ctrls['b'][1])
        
        self.optimize_axes_limits()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.active_var_idx, self.active_artist = None, None

if __name__ == "__main__":
    ui = AdvancedBCEStation(p_opt)
    plt.show()
