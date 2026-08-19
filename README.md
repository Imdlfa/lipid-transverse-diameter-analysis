# Lipid Transverse Diameter Analysis

Python workflow for calculating and comparing the transverse diameter distributions of cholesteryl esters (CEs) and triglycerides (TGs) from molecular dynamics trajectories.

## Overview

This repository provides a Python workflow for calculating molecular transverse diameters from molecular dynamics (MD) trajectories.

For each molecule, the principal molecular axis is determined by principal component analysis (PCA). The atomic coordinates are then projected onto the plane perpendicular to this axis, and the transverse diameter is calculated as the diameter of the minimum enclosing circle of the projected coordinates.

The workflow was developed to compare the transverse size distributions of cholesteryl esters and triglycerides in MD simulations.

## Requirements

* Python 3
* NumPy
* MDAnalysis
* Matplotlib

Install the required Python packages with:

```bash
pip install numpy MDAnalysis matplotlib
```

## Input Files

The script requires:

* a GROMACS topology file (`.tpr`)
* a corresponding GROMACS trajectory file (`.xtc`)

Input files and molecular selections can be specified in the user-parameter section of the script.

Example:

```python
F_TOP = "md_zedian.tpr"
F_TRJ = "md_zedian.xtc"

SEL_RN = ["MOL"]

# Cholesteryl esters
SEL_RR = [(1, 110)]

# Triglycerides
# SEL_RR = [(111, 220)]
```

## Key Parameters

The main analysis parameters can be modified directly in the script:

```python
FR_BEG = 0
FR_END = -1
FR_STEP = 5
DT_NS = 0.02

B_D = 100
B_T = 100

X_MIN = 0.0
X_MAX = 30.0

SM_D = 3.0
SM_T = 1.0
SM_N = 1
```

`FR_STEP` controls the trajectory sampling stride, whereas `DT_NS` specifies the time interval between consecutive trajectory frames.

For direct comparison between lipid species, identical histogram ranges, bin numbers and smoothing parameters should be used.

## Usage

Place the topology file, trajectory file and Python script in the same directory, or specify the corresponding file paths in the script.

Run:

```bash
python transverse_diameter.py
```

The script analyzes the selected molecules over the specified trajectory frames and calculates their transverse diameters.

## Method

For each selected molecule at each analyzed frame:

1. Periodic-boundary discontinuities within the molecule are removed by residue-based unwrapping.
2. The principal molecular axis is determined from the eigenvector corresponding to the largest eigenvalue of the coordinate covariance matrix.
3. Atomic coordinates are projected onto the two-dimensional plane perpendicular to the principal axis.
4. The minimum enclosing circle of the projected coordinates is determined.
5. The transverse diameter is calculated as twice the radius of the minimum enclosing circle.

All transverse diameters are reported in Å.

## Output

The script generates the following files:

* `raw.csv` — transverse diameter of each molecule at each analyzed frame
* `mean.csv` — mean transverse diameter at each analyzed frame
* `grid.csv` — binned time-resolved transverse diameter distribution
* `dist3d.svg` — three-dimensional visualization of the time-resolved diameter probability distribution
* `mean.svg` — mean transverse diameter as a function of simulation time

The figure format can be changed using the `FIG_FMT` parameter.

## License

This project is distributed under the MIT License.
