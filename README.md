# Feature-Based 2D Registration 

This repo contains a from-scratch, feature-based 2D image registration pipeline implemented in **C++** and **Python**, without using computer-vision libraries (e.g., OpenCV). It includes a small synthetic demo, a configuration sweep harness (via CLI flags), and technical writeups.

## Contents

### Core implementations
- `feature_registration_no_cv.cpp`  
  Main C++ implementation **with CLI flags** (detector/descriptor/model + RANSAC/matching knobs). This is the recommended entry point.
- `feature_registration_no_cv_orig.cpp`  
  Snapshot of the earlier C++ version **before** adding the CLI parsing improvements. Algorithmic core is the same; the difference is mostly `main()` argument parsing and configuration plumbing.
- `feature_based_registration_no_cv_commented.py`  
  Python implementation of a comparable pipeline, heavily commented.

### Build / project files
- `CMakeLists.txt`  
  Build configuration for the C++ binary.

### Accuracy and sweep artifacts
- `registration_config_sweep_results.csv`  
  Results for a 27-run sweep (3 detectors × 3 descriptors × 3 motion models) on the built-in synthetic pair.
- `registration_threshold_sensitivity.csv`  
  Threshold sensitivity sweep on top configurations.

### Reports / writeups
- `registration_accuracy_report.md`  
  Markdown technical report summarizing sweep accuracy.
- `registration_accuracy_report.pdf`  
  PDF version of the accuracy report.
- `feature_registration_algorithm.tex`  
  LaTeX write-up of the algorithm (flow-based, one paragraph per equation).
- `feature_registration_algorithm.pdf`  
  Compiled PDF from the LaTeX write-up.

(There may also be LaTeX build byproducts such as `.aux` and `.log`.)

---

## Algorithm overview (high-level)

Pipeline stages:

1. **Field conditioning**: smoothing + gradients  
2. **Corner scoring**: Harris / Shi–Tomasi, or FAST-9
3. **Peak selection**: non-maximum suppression / spatial pruning
4. **Orientation**: dominant local gradient direction
5. **Description**: SIFT-like 128D / normalized patch / BRIEF
6. **Matching**: distance + ratio test (+ optional mutual check)
7. **Robust fitting**: RANSAC for Similarity / Affine / Homography, with optional symmetric error
8. **Warp + diagnostics**: optional warp and error summaries

For a full derivation with equations, see `feature_registration_algorithm.tex` or `.pdf`.

---

## Build (C++)

### Requirements
- CMake >= 3.10
- A C++17 compiler (clang or g++)

### Build
```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

This produces a binary (name depends on your `CMakeLists.txt`; typically `feature_registration_no_cv`).

---

## Run (C++)

### 1) Synthetic demo (no input images)
Run with no image arguments to use the built-in synthetic pair:
```bash
./feature_registration_no_cv
```

### 2) Register two PGM images
```bash
./feature_registration_no_cv img1.pgm img2.pgm
```

### 3) Choose configuration via CLI flags
```bash
./feature_registration_no_cv --detector harris --descriptor brief --model homography
```

You can also combine with image paths:
```bash
./feature_registration_no_cv img1.pgm img2.pgm --detector shitomasi --descriptor sift --model affine
```

### Common knobs
- `--detector {harris|shitomasi|fast9}`
- `--descriptor {sift|patch|brief}`
- `--model {similarity|affine|homography}`
- `--ratio <float>` (ratio test, e.g. 0.75)
- `--iters <int>` (RANSAC iterations)
- `--ransac_thresh <float>` (pixel threshold)
- `--mutual <0|1>` (mutual best match)
- `--symmetric <0|1>` (symmetric transfer error)
- `--prosac <0|1>` (best-first sampling heuristic)

---

## Run (Python)

The Python file is self-contained (no OpenCV). Typical usage:
```bash
python feature_based_registration_no_cv_commented.py
```

If you want to feed your own images, ensure they are in a format the script supports (the C++ side uses PGM). You can adapt the Python loader accordingly.

---

## Reproducing the configuration sweep

Example sweep loops (bash pseudo-code):

```bash
for det in harris shitomasi fast9; do
  for desc in sift patch brief; do
    for model in similarity affine homography; do
      ./feature_registration_no_cv --detector $det --descriptor $desc --model $model
    done
  done
done
```

The captured results in this chat were exported into:
- `registration_config_sweep_results.csv`
- `registration_threshold_sensitivity.csv`

---

## Notes on the two C++ files

- `feature_registration_no_cv_orig.cpp` is the earlier snapshot.
- `feature_registration_no_cv.cpp` adds robust CLI parsing and config printing so runs are reproducible and do not misinterpret flags as image file names.

The registration logic itself (detectors/descriptors/matching/RANSAC) is intended to remain equivalent between them.

---

## License

If you want a license header (MIT/Apache-2.0), tell me which one to use and I will add it consistently to the sources and this README.
