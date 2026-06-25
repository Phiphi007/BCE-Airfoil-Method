import numpy as np
import math
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. 核心 BCE 几何与数学引擎
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
    CPy_h = list(p[0:7]) + [0.0]; CPy_a = list(p[7:14]); CPy_b = list(p[14:20]); gamma = p[20]
    CPx_h = [(i / 7)**gamma for i in range(8)]; CPx_a = [(i / 6)**gamma for i in range(7)]; CPx_b = [(i / 5)**gamma for i in range(6)]
    return bezier_2d(CPx_h, CPy_h), bezier_2d(CPx_a, CPy_a), bezier_2d(CPx_b, CPy_b)

def get_bce_raw_envelope(p):
   
    h_curve, a_curve, b_curve = get_bce_functions(p)
    xc_eval = np.linspace(0, 1, 2000)
    
    h_val = np.maximum(np.interp(xc_eval, h_curve[0], h_curve[1]), 1e-5) 
    alpha_val = np.radians(np.interp(xc_eval, a_curve[0], a_curve[1]))
    beta_val = np.radians(np.maximum(np.interp(xc_eval, b_curve[0], b_curve[1]), 0.1))
    
    denom = np.maximum(np.cos(beta_val)**2 - np.sin(alpha_val)**2, 1e-4)
    yc_raw = h_val * (np.sin(alpha_val) * np.cos(alpha_val)) / denom
    ay_raw = h_val * np.sin(beta_val) * np.cos(beta_val) / denom
    ax_raw = h_val * np.sin(beta_val) / np.sqrt(denom)

    # 确定原始包络线的真实左右极值 (钝头的前缘会是负数！)
    left_bounds = xc_eval - ax_raw
    right_bounds = xc_eval + ax_raw
    X_LE_raw = np.min(left_bounds)
    X_TE_raw = np.max(right_bounds)

    X_raw_eval = np.linspace(X_LE_raw, X_TE_raw, 2000)
    Y_up_raw, Y_dn_raw = np.zeros_like(X_raw_eval), np.zeros_like(X_raw_eval)
    
    for i, X in enumerate(X_raw_eval):
        mask = np.abs(X - xc_eval) <= ax_raw
        if np.any(mask):
            term = np.sqrt(np.maximum(0, 1.0 - ((X - xc_eval[mask]) / ax_raw[mask])**2))
            Y_up_raw[i] = np.max(yc_raw[mask] + ay_raw[mask] * term)
            Y_dn_raw[i] = np.min(yc_raw[mask] - ay_raw[mask] * term)

    return X_raw_eval, Y_up_raw, Y_dn_raw

# 逆向拟合归一化函数
def get_bce_airfoil_normalized(p, X_eval_targets):
    X_raw, Y_up_raw, Y_dn_raw = get_bce_raw_envelope(p)
    scale = X_raw[-1] - X_raw[0]
    X_norm = (X_raw - X_raw[0]) / scale
    Y_up_norm = Y_up_raw / scale
    Y_dn_norm = Y_dn_raw / scale
    
    Y_up_final = np.interp(X_eval_targets, X_norm, Y_up_norm)
    Y_dn_final = np.interp(X_eval_targets, X_norm, Y_dn_norm)
    Y_up_final[0] = Y_dn_final[0] = Y_up_final[-1] = Y_dn_final[-1] = 0.0
    return Y_up_final, Y_dn_final

# ==========================================
# 2. 逆向拟合 NACA 2412
# ==========================================
print("[*] 正在执行 NACA 2412 逆向拟合...")
X_target = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, 200)))
yt = 5 * 0.12 * (0.2969*np.sqrt(X_target) - 0.1260*X_target - 0.3516*X_target**2 + 0.2843*X_target**3 - 0.1015*X_target**4)
yc = np.where(X_target < 0.4, 0.02 / (0.4**2) * (2 * 0.4 * X_target - X_target**2), 
                             0.02 / (0.6**2) * (1 - 2 * 0.4 + 2 * 0.4 * X_target - X_target**2))
Y_up_target, Y_dn_target = yc + yt, yc - yt

def residual_function(p):
    try:
        Y_up_bce, Y_dn_bce = get_bce_airfoil_normalized(p, X_target)
        weight = np.ones_like(X_target); weight[:25] = 4.0 
        return np.concatenate([(Y_up_bce - Y_up_target)*weight, (Y_dn_bce - Y_dn_target)*weight])
    except: return np.ones(len(X_target)*2) * 1e3

p_init = [0.05, 0.10, 0.12, 0.15, 0.12, 0.08, 0.02] + [0.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0] + [25.0, 20.0, 15.0, 10.0, 8.0, 5.0] + [1.0]
bounds = ([1e-4]*7 + [-25.0]*7 + [2.0]*6 + [0.3], [0.4]*7 + [25.0]*7 + [60.0]*6 + [3.0])
p_opt = least_squares(residual_function, p_init, bounds=bounds, method='trf', ftol=1e-7, verbose=0).x

# ==========================================
# 3. 动画序列数据准备 (余弦分布加密)
# ==========================================
num_frames = 150
x_frames = 0.5 * (1 - np.cos(np.pi * np.linspace(1e-4, 1-1e-4, num_frames)))

h_c, a_c, b_c = get_bce_functions(p_opt)
H_arr = np.interp(x_frames, h_c[0], h_c[1])
Alpha_arr = np.radians(np.interp(x_frames, a_c[0], a_c[1]))
Beta_arr = np.radians(np.interp(x_frames, b_c[0], b_c[1]))

Denom_arr = np.cos(Beta_arr)**2 - np.sin(Alpha_arr)**2
Yc_arr = H_arr * (np.sin(Alpha_arr) * np.cos(Alpha_arr)) / Denom_arr
Ax_arr = H_arr * np.sin(Beta_arr) / np.sqrt(Denom_arr)
Ay_arr = H_arr * np.sin(Beta_arr) * np.cos(Beta_arr) / Denom_arr

X_env_raw, Yup_env_raw, Ydn_env_raw = get_bce_raw_envelope(p_opt)

# ==========================================
# 4. 画布视窗与布局初始化 (白色主调)
# ==========================================
plt.style.use('default') 
fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title('BCE Kinematics: Real Analytical Mechanism Verification')
gs = fig.add_gridspec(3, 3, width_ratios=[1.8, 1.8, 1.2], hspace=0.35)

ax3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
ax3d.set_title("3D Space: Constant Cone Height | Section Center Follows Mean Camber $Y_c(x)$", fontsize=12, weight='bold')

ax3d.set_xlim(-0.1, 1.15); ax3d.set_ylim(-0.25, 0.25); ax3d.set_zlim(-0.05, 0.25)
ax3d.set_box_aspect((1.25, 0.5, 0.3)) 

ax3d.set_xlabel("X (Chord)"); ax3d.set_ylabel("Y (Thickness)"); ax3d.set_zlabel("Z (Height)")
ax3d.view_init(elev=20, azim=-55)
ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False

xx, yy = np.meshgrid([-0.1, 1.15], [-0.25, 0.25])
surface_h = ax3d.plot_surface(xx, yy, np.zeros_like(xx), color='#e0e0e0', alpha=0.2)

ax2d = fig.add_subplot(gs[2, 0:2])
ax2d.set_title("2D Projection: Raw Space Sweep (Notice the blunt LE generation!)", fontsize=12, weight='bold')
ax2d.set_xlim(-0.1, 1.15); ax2d.set_ylim(-0.15, 0.15)
ax2d.set_aspect('equal')
ax2d.grid(True, linestyle=':', alpha=0.7)
ax2d.axhline(0, color='black', lw=0.8)

# 右侧参数变化面板
ax_h = fig.add_subplot(gs[0, 2]); ax_h.set_title("Thickness Variable $h(x)$", fontsize=10)
ax_a = fig.add_subplot(gs[1, 2]); ax_a.set_title("Camber Variable $\\alpha(x)$ (deg)", fontsize=10)
ax_b = fig.add_subplot(gs[2, 2]); ax_b.set_title("Fullness Variable $\\beta(x)$ (deg)", fontsize=10)

ax_h.plot(h_c[0], h_c[1], 'k-', lw=1.5); ax_h.set_ylim(0, 0.25)
ax_a.plot(a_c[0], a_c[1], 'm-', lw=1.5); ax_a.axhline(0, color='k', ls='--', lw=0.5); ax_a.set_ylim(-10, 15)
ax_b.plot(b_c[0], b_c[1], 'orange', lw=1.5); ax_b.set_ylim(5, 50)

for ax in [ax_h, ax_a, ax_b]:
    ax.grid(True, linestyle=':', alpha=0.5); ax.set_xlim(0, 1)

cone_surface = None
line_camber_track, = ax3d.plot([], [], [], color='red', ls='--', lw=1.5, alpha=0.8, label='Mean Camber Path $Y_c(x)$')
line_ellipse_3d, = ax3d.plot([], [], [], color='#1e3799', lw=2.5, label='Inscribed Ellipse')
line_cone_axis, = ax3d.plot([], [], [], color='black', ls='-.', lw=1.2) 

line_ellipse_2d, = ax2d.plot([], [], color='#1e3799', lw=2.5, label='Current Sweeping Section')
fill_envelope = None

tracker_h = ax_h.axvline(x=0, color='red', lw=1.5)
tracker_a = ax_a.axvline(x=0, color='red', lw=1.5)
tracker_b = ax_b.axvline(x=0, color='red', lw=1.5)
scatter_h, = ax_h.plot([], [], 'ro')
scatter_a, = ax_a.plot([], [], 'ro')
scatter_b, = ax_b.plot([], [], 'ro')

ax3d.legend(loc='upper right', fontsize=9)

# ==========================================
# 5. 固定高标准直圆锥网格生成
# ==========================================
def generate_fixed_height_pure_cone(x_c, y_c, alpha, beta):
    z_fixed_max = 0.28 
    z_loc = np.linspace(0, z_fixed_max, 18).reshape(-1, 1)
    r_loc = z_loc * np.tan(beta) 
    
    theta = np.linspace(0, 2*np.pi, 50)
    X_loc = r_loc * np.cos(theta)
    Y_loc = r_loc * np.sin(theta)
    Z_loc = np.repeat(z_loc, len(theta), axis=1)
    
    X_rot = X_loc
    Y_rot = Y_loc * np.cos(alpha) + Z_loc * np.sin(alpha)
    Z_rot = -Y_loc * np.sin(alpha) + Z_loc * np.cos(alpha)
    
    X_global = X_rot + x_c
    Y_global = Y_rot
    Z_global = Z_rot
    
    return X_global, Y_global, Z_global

# ==========================================
# 6. 动画刷新机制
# ==========================================
def update(frame):
    global cone_surface, fill_envelope, surface_h
    
    x = x_frames[frame]; h = H_arr[frame]; alpha = Alpha_arr[frame]
    beta = Beta_arr[frame]; ax = Ax_arr[frame]; ay = Ay_arr[frame]; yc = Yc_arr[frame]
    
    # ---------------- 3D 空间图层刷新 ----------------
    if cone_surface: cone_surface.remove()
    surface_h.remove()
    
    surface_h = ax3d.plot_surface(xx, yy, np.full_like(xx, h), color='#f1f2f6', alpha=0.4, zorder=1)
    
    theta_e = np.linspace(0, 2*np.pi, 100)
    E_x = x + ax * np.cos(theta_e)
    E_y = yc + ay * np.sin(theta_e)
    E_z = np.full_like(theta_e, h)
    
    line_ellipse_3d.set_data(E_x, E_y)
    line_ellipse_3d.set_3d_properties(E_z)
    
    C_X, C_Y, C_Z = generate_fixed_height_pure_cone(x, h, alpha, beta)
    cone_surface = ax3d.plot_surface(C_X, C_Y, C_Z, color='#2f3542', alpha=0.35, edgecolor='#57606f', linewidth=0.2, zorder=2)
    
    y_top_axis = 0.28 * np.sin(alpha)
    line_cone_axis.set_data([x, x], [0, y_top_axis])
    line_cone_axis.set_3d_properties([0, 0.28 * np.cos(alpha)])
    
    line_camber_track.set_data(x_frames[:frame+1], Yc_arr[:frame+1])
    line_camber_track.set_3d_properties(H_arr[:frame+1])
    
    # ---------------- 2D 扫掠图层刷新 ----------------
    if fill_envelope: fill_envelope.remove()
    
    line_ellipse_2d.set_data(E_x, E_y)
    
    mask = X_env_raw <= x
    if np.any(mask):
        fill_envelope = ax2d.fill_between(X_env_raw[mask], Ydn_env_raw[mask], Yup_env_raw[mask], color='#a4b0be', alpha=0.5)
        
    if frame == int(num_frames/2): ax2d.legend(loc='upper right', fontsize=9)
    
    # ---------------- 右侧跟踪 ----------------
    for tracker in [tracker_h, tracker_a, tracker_b]:
        tracker.set_xdata([x, x])
    scatter_h.set_data([x], [h])
    scatter_a.set_data([x], [np.degrees(alpha)])
    scatter_b.set_data([x], [np.degrees(beta)])
    
    return line_ellipse_3d, line_ellipse_2d

# ==========================================
# 7. 播放
# ==========================================
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=35, blit=False)
plt.tight_layout()
plt.show()
