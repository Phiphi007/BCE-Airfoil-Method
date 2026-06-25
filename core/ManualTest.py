import numpy as np
import math
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 核心数学与 BCE 几何生成器
# ==========================================
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
    
    h_curve = bezier_2d(CPx_h, CPy_h)
    a_curve = bezier_2d(CPx_a, CPy_a)
    b_curve = bezier_2d(CPx_b, CPy_b)
    
    ctrl_pts = {'h': (CPx_h, CPy_h), 'a': (CPx_a, CPy_a), 'b': (CPx_b, CPy_b)}
    return h_curve, a_curve, b_curve, ctrl_pts

def get_bce_airfoil(p, X_eval_targets):
   
    h_curve, a_curve, b_curve, _ = get_bce_functions(p)
    t_eval = np.linspace(0, 1, 2000)
    
    h_val = np.maximum(np.interp(t_eval, h_curve[0], h_curve[1]), 1e-5) 
    alpha_val = np.radians(np.interp(t_eval, a_curve[0], a_curve[1]))
    beta_val = np.radians(np.maximum(np.interp(t_eval, b_curve[0], b_curve[1]), 0.1))
    
    denom = np.maximum(np.cos(beta_val)**2 - np.sin(alpha_val)**2, 1e-4)
    
    xc_raw = t_eval
    yc_raw = h_val * (np.sin(alpha_val) * np.cos(alpha_val)) / denom
    ax_raw = h_val * np.sin(beta_val) / np.sqrt(denom)
    ay_raw = h_val * np.sin(beta_val) * np.cos(beta_val) / denom

    left_bounds, right_bounds = xc_raw - ax_raw, xc_raw + ax_raw
    idx_LE, idx_TE = np.argmin(left_bounds), np.argmax(right_bounds)
    X_LE_raw, Y_LE_raw = left_bounds[idx_LE], yc_raw[idx_LE]
    X_TE_raw, Y_TE_raw = right_bounds[idx_TE], yc_raw[idx_TE]
    
    chord = np.sqrt((X_TE_raw - X_LE_raw)**2 + (Y_TE_raw - Y_LE_raw)**2)
    theta = -np.arctan2(Y_TE_raw - Y_LE_raw, X_TE_raw - X_LE_raw)

    X_raw_eval = X_LE_raw + (X_TE_raw - X_LE_raw) * t_eval
    Y_up_raw, Y_dn_raw = np.zeros_like(X_raw_eval), np.zeros_like(X_raw_eval)
    
    for i, X in enumerate(X_raw_eval):
        mask = np.abs(X - xc_raw) <= ax_raw
        if np.any(mask):
            term = np.sqrt(np.maximum(0, 1.0 - ((X - xc_raw[mask]) / ax_raw[mask])**2))
            Y_up_raw[i] = np.max(yc_raw[mask] + ay_raw[mask] * term)
            Y_dn_raw[i] = np.min(yc_raw[mask] - ay_raw[mask] * term)

    def transform(x, y):
        xs, ys = x - X_LE_raw, y - Y_LE_raw
        return (xs * np.cos(theta) - ys * np.sin(theta)) / chord, (xs * np.sin(theta) + ys * np.cos(theta)) / chord

    X_up_norm, Y_up_norm = transform(X_raw_eval, Y_up_raw)
    X_dn_norm, Y_dn_norm = transform(X_raw_eval, Y_dn_raw)
    X_cam_norm, Y_cam_norm = transform(xc_raw, yc_raw) # 转换中弧线

    idx_up, idx_dn = np.argsort(X_up_norm), np.argsort(X_dn_norm)
    idx_cam = np.argsort(X_cam_norm)
    
    Y_up_final = np.interp(X_eval_targets, X_up_norm[idx_up], Y_up_norm[idx_up])
    Y_dn_final = np.interp(X_eval_targets, X_dn_norm[idx_dn], Y_dn_norm[idx_dn])
    Y_cam_final = np.interp(X_eval_targets, X_cam_norm[idx_cam], Y_cam_norm[idx_cam])
    
    return Y_up_final, Y_dn_final, Y_cam_final

# ==========================================
# 2. 目标文件读取与脏数据清洗
# ==========================================
def get_naca2412():
    X_t = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, 200)))
    t = 0.12
    yt = 5 * t * (0.2969*np.sqrt(X_t) - 0.1260*X_t - 0.3516*X_t**2 + 0.2843*X_t**3 - 0.1015*X_t**4)
    yc = np.where(X_t < 0.4, 0.02 / (0.4**2) * (2 * 0.4 * X_t - X_t**2), 
                             0.02 / (0.6**2) * (1 - 2 * 0.4 + 2 * 0.4 * X_t - X_t**2))
    return X_t, yc + yt, yc - yt

def process_dat_file(filename, num_eval=200):
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
        print(f"[*] 检测到钝后缘 (厚度: {te_thickness:.5f})，自动执行切线相交延长闭合...")
        p_u1, p_u0 = upper_raw[-2], upper_raw[-1]
        p_l1, p_l0 = lower_raw[-2], lower_raw[-1]
        m_u = (p_u0[1] - p_u1[1]) / (p_u0[0] - p_u1[0] + 1e-8)
        m_l = (p_l0[1] - p_l1[1]) / (p_l0[0] - p_l1[0] + 1e-8)
        if abs(m_u - m_l) > 1e-5:
            X_int = (m_u * p_u0[0] - m_l * p_l0[0] - p_u0[1] + p_l0[1]) / (m_u - m_l)
            Y_int = m_u * (X_int - p_u0[0]) + p_u0[1]
            virtual_TE = np.array([X_int, Y_int])
            upper_raw = np.vstack([upper_raw, virtual_TE])
            lower_raw = np.vstack([lower_raw, virtual_TE])
        else:
            virtual_TE = np.array([p_u0[0] + 0.01, (p_u0[1]+p_l0[1])/2])
            upper_raw = np.vstack([upper_raw, virtual_TE])
            lower_raw = np.vstack([lower_raw, virtual_TE])
    else:
        print("[*] 检查通过：输入翼型为尖后缘，无需处理。")
        virtual_TE = TE_up

    LE = upper_raw[0]
    dx, dy = virtual_TE[0] - LE[0], virtual_TE[1] - LE[1]
    chord = np.sqrt(dx**2 + dy**2)
    theta = -np.arctan2(dy, dx)
    
    def transform(pts):
        x_s, y_s = pts[:, 0] - LE[0], pts[:, 1] - LE[1]
        return (x_s * np.cos(theta) - y_s * np.sin(theta)) / chord, (x_s * np.sin(theta) + y_s * np.cos(theta)) / chord

    upper_norm_x, upper_norm_y = transform(upper_raw)
    lower_norm_x, lower_norm_y = transform(lower_raw)

    X_target = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_eval)))
    Y_up_target = interp1d(upper_norm_x, upper_norm_y, kind='cubic', fill_value='extrapolate')(X_target)
    Y_dn_target = interp1d(lower_norm_x, lower_norm_y, kind='cubic', fill_value='extrapolate')(X_target)
    
    Y_up_target[0] = Y_dn_target[0] = Y_up_target[-1] = Y_dn_target[-1] = 0.0
    return X_target, Y_up_target, Y_dn_target

# ==========================================
# 3. 交互流程与拟合
# ==========================================
print("="*60)
print("  BCE 翼型逆向拟合与交互微调平台 (Human-in-the-loop)")
print("="*60)
user_input = input("请输入目标翼型的 dat 文件名 (留空回车则默认 NACA2412): ").strip()

if user_input == "":
    print("\n[*] 使用基准翼型: NACA 2412")
    X_target, Y_up_target, Y_dn_target = get_naca2412()
    title_str = "NACA 2412 (Default Baseline)"
else:
    if os.path.exists(user_input):
        print(f"\n[*] 正在读取并清洗文件: {user_input} ...")
        X_target, Y_up_target, Y_dn_target = process_dat_file(user_input)
        title_str = f"Fitted Target: {user_input}"
    else:
        print(f"\n[!] 错误: 找不到文件 '{user_input}'。回退至 NACA 2412。")
        X_target, Y_up_target, Y_dn_target = get_naca2412()
        title_str = "NACA 2412 (Fallback)"

def residual_function(p):
    try:
        Y_up_bce, Y_dn_bce, _ = get_bce_airfoil(p, X_target)
        weight = np.ones_like(X_target)
        weight[:25] = 4.0 
        return np.concatenate([(Y_up_bce - Y_up_target)*weight, (Y_dn_bce - Y_dn_target)*weight])
    except:
        return np.ones(len(X_target)*2) * 1e3

print("\n[*] 启动 21 维 BCE 寻优算法 (Least Squares TRF) ...")
p_init = [0.05, 0.10, 0.12, 0.15, 0.12, 0.08, 0.02] + [0.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0] + [25.0, 20.0, 15.0, 10.0, 8.0, 5.0] + [1.0]
bounds = ([1e-4]*7 + [-25.0]*7 + [2.0]*6 + [0.3], [0.4]*7 + [25.0]*7 + [60.0]*6 + [3.0])

res = least_squares(residual_function, p_init, bounds=bounds, method='trf', ftol=1e-8, verbose=0)
p_opt = res.x
Y_up_opt, Y_dn_opt, Y_cam_opt = get_bce_airfoil(p_opt, X_target)
max_err = np.max(np.abs(np.concatenate([Y_up_opt-Y_up_target, Y_dn_opt-Y_dn_target])))
print(f"[+] 拟合成功！最大归一化残差: {max_err:.6f}")
print("\n[*] 即将开启交互式气动设计面板 ...")

# ==========================================
# 4. 高级交互可视化类
# ==========================================
class InteractiveBCE:
    def __init__(self, p_base):
        self.p = p_base.copy()
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('BCE Interactive Aerodynamic Design')
        
        self.ax1 = self.fig.add_subplot(2, 2, 1) 
        self.ax2 = self.fig.add_subplot(2, 2, 2) 
        self.ax3 = self.fig.add_subplot(2, 2, 3) 
        self.ax4 = self.fig.add_subplot(2, 2, 4) 
        
        self.fig.suptitle(f"{title_str}\n(Drag Red Points to Edit | Initial Max Err: {max_err:.5f})", fontsize=16, fontweight='bold')

        # --- 视窗 1: 翼型与中弧线 ---
        self.ax1.set_title("1. Airfoil (Gray: Baseline | Colored: Live Edit)", fontsize=11)
        self.ax1.plot(X_target, Y_up_target, color='lightgray', linestyle='--', linewidth=3, label='Target Baseline')
        self.ax1.plot(X_target, Y_dn_target, color='lightgray', linestyle='--', linewidth=3)
        self.ax1.plot([0,1], [0,0], 'k-', linewidth=0.5)
        
        self.line_up, = self.ax1.plot(X_target, Y_up_opt, 'b-', linewidth=2, label='Upper')
        self.line_dn, = self.ax1.plot(X_target, Y_dn_opt, 'g-', linewidth=2, label='Lower')
        self.line_cam, = self.ax1.plot(X_target, Y_cam_opt, 'r-', linewidth=1.5, label='Camber Line') # 中弧线
        self.ax1.axis('equal')
        self.ax1.grid(True, linestyle=':', alpha=0.6)
        self.ax1.legend(loc='upper right')

        h_curve, a_curve, b_curve, ctrls = get_bce_functions(self.p)
        
        # --- 视窗 2: 厚度 ---
        self.ax2.set_title("2. Thickness h(x)", fontsize=11)
        self.line_h, = self.ax2.plot(h_curve[0], h_curve[1], 'k-', linewidth=2)
        # 增加控制多边形连线
        self.poly_h, = self.ax2.plot(ctrls['h'][0], ctrls['h'][1], 'r-', linewidth=1, alpha=0.5) 
        self.pts_h, = self.ax2.plot(ctrls['h'][0][:-1], ctrls['h'][1][:-1], 'ro', markersize=10, picker=True, pickradius=10)
        self.ax2.plot(ctrls['h'][0][-1], ctrls['h'][1][-1], 'ko', markersize=6)
        self.ax2.set_ylim(-0.05, 0.4)
        self.ax2.grid(True, linestyle=':', alpha=0.6)

        # --- 视窗 3: 弯度 ---
        self.ax3.set_title("3. Camber $\\alpha(x)$", fontsize=11)
        self.line_a, = self.ax3.plot(a_curve[0], a_curve[1], 'm-', linewidth=2)
        self.poly_a, = self.ax3.plot(ctrls['a'][0], ctrls['a'][1], 'r-', linewidth=1, alpha=0.5)
        self.pts_a, = self.ax3.plot(ctrls['a'][0], ctrls['a'][1], 'ro', markersize=10, picker=True, pickradius=10)
        self.ax3.axhline(0, color='k', linestyle='--', linewidth=1)
        self.ax3.set_ylim(-25, 25)
        self.ax3.grid(True, linestyle=':', alpha=0.6)

        # --- 视窗 4: 顶角 ---
        self.ax4.set_title("4. Fullness $\\beta(x)$", fontsize=11)
        self.line_b, = self.ax4.plot(b_curve[0], b_curve[1], 'orange', linewidth=2)
        self.poly_b, = self.ax4.plot(ctrls['b'][0], ctrls['b'][1], 'r-', linewidth=1, alpha=0.5)
        self.pts_b, = self.ax4.plot(ctrls['b'][0], ctrls['b'][1], 'ro', markersize=10, picker=True, pickradius=10)
        self.ax4.set_ylim(0, 60)
        self.ax4.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

        self.active_var_idx = None  
        self.active_artist = None   

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes is None: return
        for ax, pts_artist, offset in zip(
                [self.ax2, self.ax3, self.ax4], 
                [self.pts_h, self.pts_a, self.pts_b], 
                [0, 7, 14]):
            if event.inaxes == ax:
                cont, ind = pts_artist.contains(event)
                if cont:
                    self.active_var_idx = offset + ind["ind"][0] 
                    self.active_artist = pts_artist
                    break

    def on_motion(self, event):
        if self.active_var_idx is None or event.inaxes is None: return
        
        new_y = event.ydata
        if self.active_var_idx < 7:       new_y = max(0.001, new_y)
        elif self.active_var_idx >= 14:   new_y = max(2.0, min(80.0, new_y))
            
        self.p[self.active_var_idx] = new_y
        
        h_curve, a_curve, b_curve, ctrls = get_bce_functions(self.p)
        try:
            Y_up_new, Y_dn_new, Y_cam_new = get_bce_airfoil(self.p, X_target)
        except: return 

        self.line_up.set_ydata(Y_up_new)
        self.line_dn.set_ydata(Y_dn_new)
        self.line_cam.set_ydata(Y_cam_new) # 实时刷新中弧线
        
        # 刷新厚度图
        self.line_h.set_ydata(h_curve[1])
        self.pts_h.set_ydata(ctrls['h'][1][:-1]) 
        self.poly_h.set_ydata(ctrls['h'][1])
        
        # 刷新弯度图
        self.line_a.set_ydata(a_curve[1])
        self.pts_a.set_ydata(ctrls['a'][1])
        self.poly_a.set_ydata(ctrls['a'][1])
        
        # 刷新顶角图
        self.line_b.set_ydata(b_curve[1])
        self.pts_b.set_ydata(ctrls['b'][1])
        self.poly_b.set_ydata(ctrls['b'][1])
        
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.active_var_idx = None
        self.active_artist = None

if __name__ == "__main__":
    ui = InteractiveBCE(p_opt)
    plt.show()
