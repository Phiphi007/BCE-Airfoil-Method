\# BCE (Brush-Cone Envelope) Airfoil Parameterization Method



\[!\[License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

\[!\[Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.14-green.svg)](https://www.python.org/)



> 💡 \*This method was born out of a sudden personal mathematical inspiration. / 本方法源于个人的一个数学突发奇想。\*



\[English](#english) | \[中文](#中文)



\---



<a name="english"></a>

\## 🇬🇧 English Description



\*\*BCE (Brush-Cone Envelope)\*\* is a highly decoupled 2D airfoil parameterization methodology based on 3D spatial geometry abstraction. It focuses entirely on building an explicit, closed-form algebraic mapping between an airfoil's physical traits (Thickness \& Camber) and its mathematical generation manifold.

This project is tested and compatible with \*\*Python 3.14\*\*.

\#### Required Packages

The core mathematical engine, interactive UIs, and 3D geometric visualizations rely on standard scientific Python libraries and PyVista. You can install them via `pip`:

```bash

pip install numpy scipy matplotlib pyvista



\### 1. The Mathematical Core: Spatial Abstraction

The BCE geometry is inspired by slicing an inverted moving cone in 3D space:

1\. Imagine an inverted cone whose half-angle is defined by a fullness function $\\beta(x)$.

2\. The cone tilts by an angle $\\alpha(x)$ (dictating the camber).

3\. A spatial cutting plane is placed at height $h(x)$ (dictating the thickness).

4\. The \*\*envelope of the projected ellipses\*\* formed by this moving slice yields a strictly smooth and continuous 2D airfoil.



The skeletal camber $Y\_c(x)$ and half-thickness $Z\_t(x)$ are strictly mapped by explicit closed-form algebraic expressions:

$$Y\_c(x) = h(x) \\cdot \\frac{\\sin\\alpha(x)\\cos\\alpha(x)}{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}$$

$$Z\_t(x) = h(x) \\cdot \\frac{\\sin\\beta(x)}{\\sqrt{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}}$$



\### 2. Hard-Constrained Deformation Mechanism

Because $Y\_c(x)$ and $Z\_t(x)$ share a highly symmetric nonlinear manifold topology, BCE allows for \*\*absolute rigid locking of specific geometric features\*\* via analytical inverse compensation:

\* \*\*Locking Thickness ($Z\_t$ Frozen)\*\*: When structural volume or thickness constraints must remain unchanged during geometric morphing, we enforce $Z\_t(x) = Z\_{target}(x)$. The height function $h(x)$ is analytically solved as a compensator:

&#x20; $$h(x) = Z\_{target}(x) \\cdot \\frac{\\sqrt{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}}{\\sin\\beta(x)}$$

&#x20; The user can drastically bend the camber profile $\\alpha(x)$ while the thickness envelope is rigidly held by the algebraic manifold without any non-physical distortions.

\* \*\*Locking Camber ($Y\_c$ Frozen)\*\*: The thickness distribution can be scaled or optimized freely while the exact camber skeleton remains strictly intact to freeze lift characteristics:

&#x20; $$h(x) = Y\_{target}(x) \\cdot \\frac{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}{\\sin\\alpha(x)\\cos\\alpha(x)}$$



> 💡 \*Note on Implementation: To maximize numerical robustness during large-scale interactive morphing, the backend core code directly bypasses control point pseudo-inverse fitting and implements this rigid boundary via an explicit geometric override on the spatial ellipse generation level.\*



\### 3. File Structure \& Playground

All functional scripts are located in the `core/` folder:

\* `Animation.py`: A 3D mechanism visualizer demonstrating how the moving cone generates the airfoil envelope.

\* `ManualTest.py`: Reverse-fitting module. Enter a `.dat` filename to fit it into BCE variables (Press `Enter` directly to load the default NACA 2412).

\* `DecouplingDeform.py`: An interactive UI demonstrating the "Free Morphing" and the "Rigidly Constrained Morphing" mechanisms.

\* `DeformXfoil.py`:  A primitive, basic prototype script integrating the morphing engine with xfoil.exe. It provides a rough, interactive demonstration of tracking aerodynamic performance changes ($C\_L$, $L/D$) simultaneously as the shape mutates. Feel free to play around and hack it!



\### 📬 Contact

\* \*\*Author\*\*: Phiphi007

\* \*\*Email\*\*: phiphihyf@gmail.com



\---



<a name="中文"></a>

\## 🇨🇳 中文说明



\*\*BCE (Brush-Cone Envelope)\*\* 是一种基于三维空间几何投影抽象、具备完全显式特征解耦能力的二维翼型参数化数学流形方法。该方法的核心在于建立了一套“厚度场”与“弯度场”之间绝对闭式的代数映射体系。

本程序已在Python 3.14 环境下进行测试，均可完美兼容运行。

\####  必需的 Python 库

核心数学流形计算、交互式 UI 界面以及 3D 几何可视化渲染，依赖于基础的科学计算库与 PyVista。你可以通过以下命令一键安装：

```bash

pip install numpy scipy matplotlib pyvista



\### 1. 数学流形抽象核心

BCE 方法的几何灵感来源于三维空间中的随动圆锥切割投影：

1\. 假设一个倒置圆锥，其半顶角定义为饱满度函数 $\\beta(x)$。

2\. 圆锥绕其顶点发生角度为 $\\alpha(x)$ 的倾斜偏转（映射弯度）。

3\. 引入空间切割高度函数 $h(x)$（映射厚度）。

4\. 在高度 $h(x)$ 处对该圆锥进行切片，该随动椭圆族在 $x-y$ 投影面上的\*\*解析外包络线\*\*，即构成平滑的翼型型面。



中弧线骨架 $Y\_c(x)$ 与半厚度分布 $Z\_t(x)$ 满足以下完全闭式的显式代数映射：

$$Y\_c(x) = h(x) \\cdot \\frac{\\sin\\alpha(x)\\cos\\alpha(x)}{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}$$

$$Z\_t(x) = h(x) \\cdot \\frac{\\sin\\beta(x)}{\\sqrt{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}}$$



\### 2. 刚性约束变形数学机制

得益于高度对称的非线性代数拓扑，BCE 方法能够通过解析解的反解自适应补偿，实现某一几何特征的“绝对刚性锁死”：

\* \*\*厚度分布绝对锁定 (Lock Zt)\*\*：在变体机构变形或施加严格的结构容积约束时，令 $Z\_t(x) = Z\_{target}(x)$。此时高度函数 $h(x)$ 被直接作为自适应补偿项进行解析反解：

&#x20; $$h(x) = Z\_{target}(x) \\cdot \\frac{\\sqrt{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}}{\\sin\\beta(x)}$$

&#x20; 此时用户可以随意大尺度拖拽弯度分布 $\\alpha(x)$，而翼型截面的厚度包络被代数流形绝对锁死，不会发生任何传统参数化方法中常见的非物理畸变（如弯度增大导致翼型意外变薄）。

\* \*\*弯度分布绝对锁定 (Lock Yc)\*\*：在保持原有升力中弧线特性严格不被破坏（从而冻结基本气动升力特征）的前提下，通过以下补偿方程，可自由修形和缩放厚度分布场：

&#x20; $$h(x) = Y\_{target}(x) \\cdot \\frac{\\cos^2\\beta(x) - \\sin^2\\alpha(x)}{\\sin\\alpha(x)\\cos\\alpha(x)}$$



> 💡 \*工程实现注：为了在 UI 交互界面进行大尺度正负拓拽时获得极致的数值稳定性，后端核心代码并未强行反解控制点，而是采用显式空间几何覆盖（Explicit Geometric Override）的方式直接作用于底层椭圆包络线生成，保证了极端变形下曲线的绝对平滑。\*



\### 3. 代码把玩指南 (Playground)

核心代码全部位于 `core/` 文件夹下，包含几个极具可视化反馈的交互终端：

\* `Animation.py`：3D 数学机制动画引擎，直观展示圆锥切片生成翼型包络的空间几何过程。

\* `ManualTest.py`：逆向拟合测试站。运行后输入自带的 `.dat` 文件名即可将其反解为 BCE 特征（直接敲击回车默认载入 NACA 2412）。

\* `DecouplingDeform.py`：交互式大画板。可以在 UI 面板中直观体验“自由变形”与“锁定特征约束变形”的手感差异。

\* `DeformXfoil.py`：这是一个比较简陋的初始联动测试版本，尝试将几何大变形与底层的 xfoil.exe 驱动总线结合。能够在调节外形的同时，粗糙地同步展示气动升力系数（$C\_L$）和升阻比效率（$L/D$）的变化曲线，供感兴趣的极客探索和魔改。需要将xfoil.exe放在工作目录下。



\### 📬 联系方式

\* \*\*Author\*\*: Phiphi007

\* \*\*Email\*\*: phiphihyf@gmail.com

