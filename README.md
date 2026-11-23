# Jansen Mechanism Link Dimensional Synthesis for Human Gait

An automated kinematic synthesis pipeline that **back-calculates the link lengths of a Jansen leg mechanism** to accurately fit a target **human foot gait trajectory**.

Theo Jansen's iconic 11 "holy numbers" were originally designed for kinetic beach sculptures. This project formulates mechanism dimensional synthesis as an inverse-kinematics optimization problem to synthesize linkage dimensions ($a$ through $m$) tailored to human walking biomechanics, stance flatness, and ground clearance.

---

## Features

- **Geometric Path Synthesis**: Uses arc-length reparameterization to decouple non-linear crank velocity from trajectory geometry, achieving sub-centimeter geometric RMSE.
- **Closed-Form 2D Alignment (Kabsch Algorithm)**: Analytically computes optimal rigid rotation and cyclic phase matching, eliminating rotational local minima and speeding up optimization to ~5–10 seconds.
- **Multi-Stage Optimization**: Combines global Differential Evolution (DE) with Nelder-Mead simplex search and Powell directional polishing.
- **Biomechanical Analysis**: Evaluates ground stance flatness ($\sigma_y$), stance duty factor (% of cycle in ground support), and 4-bar transmission angles to prevent toggle jamming.
- **Visual Outputs**: Exports publication-quality comparison figures and an animated GIF showing the moving planar linkage and traced foot path.

---

## Project Structure

```
JesenMech/
├── data/
│   └── human_trajectory.csv       # Standard human foot gait trajectory landmarks
├── src/
│   ├── main.py                    # Main execution script & CLI entrypoint
│   ├── positional_analysis.py     # Kinematic Center of Mass (CoM) positional analysis
│   ├── jansen.py                  # Forward kinematics and joint solver (O, Y, Z, X, W, V, U, T)
│   ├── optimizer.py               # Optimization engine, Kabsch alignment, and synthesis
│   ├── trajectory.py              # Trajectory loader with periodic cubic spline interpolation
│   ├── animate.py                 # Generates kinematic linkage animation GIF
│   ├── animate_pygame.py          # Interactive Pygame desktop simulator
│   └── animate_gui.py             # Zero-dependency Tkinter desktop visualizer
├── link_centers_of_mass.csv       # Output: CoM coordinates of each link vs crank angle
├── link_com_trajectories.png      # Output: 2D CoM trajectories & X/Y displacement plots
├── back_calculated_links.csv      # Output: Synthesized link dimensions and ratios
├── optimized_trajectory.csv       # Output: Coordinates traced by synthesized foot
├── trajectory_comparison.png      # Output: 2-Panel trajectory & stance comparison plot
├── mechanism_animation.gif        # Output: Animated walking linkage
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## Prerequisites & Installation

### 1. Clone the repository
```bash
git clone https://github.com/vedantdroid/JesenMech.git
cd JesenMech
```

### 2. Set up a Python Virtual Environment
Requires **Python 3.8+**:
```bash
python3 -m venv venv
source venv/bin/activate
```
*(On Windows: `venv\Scripts\activate`)*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Default Run (Recommended)
Runs the optimization on `data/human_trajectory.csv` in geometric shape mode and generates plots and animation:
```bash
python3 src/main.py
```

### 2. Synthesis Modes
The pipeline supports three optimization modes via `--mode`:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **`shape`** *(default)* | `python3 src/main.py --mode shape` | Arc-length geometric path matching (lowest spatial RMSE: ~1.3% error). |
| **`synchronized`** | `python3 src/main.py --mode synchronized` | Point-to-point crank angle time-synchronized matching. |
| **`balanced`** | `python3 src/main.py --mode balanced` | Joint optimization: 70% geometric shape + 30% timing synchronization. |

### 3. Custom Human Trajectory
To synthesize links for your own trajectory CSV:
```bash
python3 src/main.py --input path/to/your_trajectory.csv
```
> **CSV Format**: The CSV only needs `x` and `y` columns containing 2D points along the foot path. The loader automatically handles sparse landmarks or dense points using periodic cubic spline interpolation.

---

## Positional Analysis (Link Centers of Mass)

To compute the $(X, Y)$ coordinates of the **Center of Mass (CoM)** for each moving link with the **crank pivot axis as the origin $(0, 0)$**:

### 1. Run 360° Sweep & Generate Plots
```bash
python3 src/positional_analysis.py
```
This generates:
- [`link_centers_of_mass.csv`](file:///home/vedantdroid/JesenMech/link_centers_of_mass.csv): 360-row table ($1^\circ$ resolution) with $(x, y)$ for all links and joints.
- [`link_com_trajectories.png`](file:///home/vedantdroid/JesenMech/link_com_trajectories.png): 3-panel figure showing 2D spatial CoM trajectories and horizontal/vertical displacement curves.

### 2. Query a Single Crank Angle
```bash
python3 src/positional_analysis.py --angle 45.0
```
Outputs a real-time console readout of all $(X, Y)$ Centers of Mass at $\theta = 45^\circ$.

---

## Output Files

When `src/main.py` finishes, it generates:

1. **`back_calculated_links.csv`**:
   Table containing all 13 link lengths:
   - Base Jansen dimension (mm)
   - Initial scaled baseline (mm)
   - Synthesized / back-calculated length (mm)
   - Percentage change (%)
   - Dimensional ratio relative to crank $m$
2. **`optimized_trajectory.csv`**:
   The $(x, y)$ coordinates of the synthesized foot locus aligned to the human target.
3. **`trajectory_comparison.png`**:
   High-resolution 2-panel figure:
   - **Panel 1**: 2D Sagittal Plane Foot Trajectory Comparison (Target Human vs Synthesized Jansen vs Baseline Jansen).
   - **Panel 2**: Ground Stance Flatness Profile & Vertical Clearance vs Horizontal Stride.
4. **`mechanism_animation.gif`**:
   Full 50-frame animated GIF showing all bars, pivots, the triangular lower leg ($W$-$U$-$T$), and foot trace through $360^\circ$ of crank rotation.

---

## Mechanism Nomenclature Reference

The Jansen leg mechanism consists of 8 moving links, 1 driving crank, 2 fixed frame pivots, and a triangular foot:

| Symbol | Description | Connected Joints |
| :---: | :--- | :---: |
| **$m$** | Driving crank radius | $O \to Z$ |
| **$a$** | Ground pivot horizontal offset | $O \to Y$ ($X$ component) |
| **$l$** | Ground pivot vertical offset | $O \to Y$ ($Y$ component) |
| **$j$** | Upper crank coupler | $Z \to X$ |
| **$b$** | Upper rocker | $Y \to X$ |
| **$k$** | Lower crank coupler | $Z \to W$ |
| **$c$** | Lower rocker | $Y \to W$ |
| **$d$** | Intermediate rocker | $Y \to V$ |
| **$e$** | Intermediate coupler | $X \to V$ |
| **$f$** | Lower connecting link | $V \to U$ |
| **$g$** | Lower connecting link | $W \to U$ |
| **$h$** | Lower leg triangle side | $U \to T$ (foot) |
| **$i$** | Lower leg triangle side | $W \to T$ (foot) |

- **$O$**: Fixed crank pivot `(0, 0)`
- **$Y$**: Fixed frame pivot `(-a, -l)`
- **$Z$**: Rotating crank tip
- **$T$**: Foot contact point
