# Feature-Based 2D Registration Evaluation

## Summary

- Runs: **27**
- Successful: **24** (88.9%)
- Corner error vs ground truth (px): min **0.474**, median **1.163**, max **3.141**
- Mean inlier reprojection error (px): min **0.933**, median **1.277**, max **1.724**

## Setup

- Data: built-in synthetic pair produced by the C++ demo (single pair).
- Sweep: 27 combinations (3 detectors x 3 descriptors x 3 motion models).
- Fixed hyperparameters: ratio=0.75, iters=700, mutual=1, symmetric=1, PROSAC=1, RANSAC threshold=3.2 px.

## Metrics

- **Corner error (px)**: Euclidean error between projected image corners under the estimated transform and the known ground-truth transform.
- **Mean inlier reprojection error (px)**: average reprojection residual over inlier correspondences selected by RANSAC.
- **Success**: program exits with code 0 (enough matches, RANSAC converged).

## Results by component choice

### Motion model

| Model | Runs | Successful | Success | Corner med | Corner min | Corner max | Mean inlier err med | Inlier ratio med | Matches med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| homography | 9 | 8 | 88.9% | 0.764 | 0.474 | 3.141 | 1.277 | 0.9401162790697674 | 28.0 |
| affine | 9 | 8 | 88.9% | 1.075 | 0.923 | 1.851 | 1.432 | 0.9559108527131783 | 28.0 |
| similarity | 9 | 8 | 88.9% | 1.962 | 0.977 | 2.535 | 0.969 | 0.20016611295681053 | 28.0 |

### Detector

| Detector | Runs | Successful | Success | Corner med | Corner min | Corner max | Mean inlier err med | Inlier ratio med | Matches med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| harris | 9 | 9 | 100.0% | 0.933 | 0.474 | 2.148 | 1.143 | 0.8928571428571429 | 28.0 |
| shitomasi | 9 | 9 | 100.0% | 1.094 | 0.718 | 1.988 | 1.355 | 0.9534883720930232 | 43.0 |
| fast9 | 9 | 6 | 66.7% | 2.193 | 1.361 | 3.141 | 1.499 | 0.72 | 18.0 |

### Descriptor

| Descriptor | Runs | Successful | Success | Corner med | Corner min | Corner max | Mean inlier err med | Inlier ratio med | Matches med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sift | 9 | 9 | 100.0% | 1.276 | 0.555 | 3.141 | 1.278 | 0.8571428571428571 | 28.0 |
| patch | 9 | 9 | 100.0% | 1.361 | 0.529 | 2.535 | 1.296 | 0.72 | 25.0 |
| brief | 9 | 6 | 66.7% | 1.075 | 0.474 | 1.988 | 1.205 | 1.0 | 34.5 |

## Best configurations

Top-10 by corner error (lower is better):

| Detector | Descriptor | Model | Matches | Inliers | Inlier ratio | Mean inlier err (px) | Corner err (px) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| harris | brief | homography | 28 | 28 | 100.0% | 1.143 | 0.474 |
| harris | patch | homography | 20 | 19 | 95.0% | 1.109 | 0.529 |
| harris | sift | homography | 28 | 25 | 89.3% | 1.152 | 0.555 |
| shitomasi | patch | homography | 43 | 40 | 93.0% | 1.276 | 0.718 |
| shitomasi | sift | homography | 48 | 46 | 95.8% | 1.278 | 0.810 |
| harris | sift | affine | 28 | 24 | 85.7% | 0.988 | 0.923 |
| harris | patch | affine | 20 | 20 | 100.0% | 1.296 | 0.933 |
| harris | brief | similarity | 28 | 6 | 21.4% | 1.064 | 0.977 |
| shitomasi | patch | affine | 43 | 41 | 95.3% | 1.449 | 1.026 |
| shitomasi | brief | affine | 41 | 41 | 100.0% | 1.541 | 1.072 |

## Failure modes observed

The following combinations failed (exit code != 0):

| Detector | Descriptor | Model | Exit code |
| --- | --- | --- | --- |
| fast9 | brief | similarity | 2 |
| fast9 | brief | affine | 2 |
| fast9 | brief | homography | 2 |

## RANSAC threshold sensitivity

Sweep on the three best configurations using thresholds 2.0 / 3.2 / 5.0 px:

| Detector | Descriptor | Model | Thresh (px) | Matches | Inliers | Inlier ratio | Mean inlier err (px) | Corner err (px) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| harris | brief | homography | 2.0 | 28.0 | 24.0 | 85.7% | 1.007 | 1.222 |
| harris | brief | homography | 3.2 | 28.0 | 28.0 | 100.0% | 1.143 | 0.474 |
| harris | brief | homography | 5.0 | 28.0 | 28.0 | 100.0% | 1.143 | 0.474 |
| harris | patch | homography | 2.0 | 20.0 | 16.0 | 80.0% | 0.907 | 0.476 |
| harris | patch | homography | 3.2 | 20.0 | 19.0 | 95.0% | 1.109 | 0.529 |
| harris | patch | homography | 5.0 | 20.0 | 20.0 | 100.0% | 1.272 | 0.898 |
| harris | sift | homography | 2.0 | 28.0 | 21.0 | 75.0% | 0.757 | 1.292 |
| harris | sift | homography | 3.2 | 28.0 | 25.0 | 89.3% | 1.152 | 0.555 |
| harris | sift | homography | 5.0 | 28.0 | 26.0 | 92.9% | 1.278 | 0.899 |

## Practical recommendations (based on this run)

- If the transform can include perspective, **Homography** delivered the lowest corner error for the strongest configs.
- **Harris** and **Shi-Tomasi** were stable across all descriptors in this sweep.
- **FAST9 + BRIEF** was brittle here (all three models failed). Pairing FAST9 with SIFT or Patch worked.
- RANSAC threshold interacts with the descriptor: BRIEF and SIFT preferred ~3.2 px in this run, Patch preferred 2.0 px.

## Artifacts

- Charts: `success_rate_by_detector.png`, `success_rate_by_descriptor.png`, `corner_error_by_model.png`, `corner_error_top10.png`, `ransac_threshold_sensitivity.png`
- CSV sources: `registration_config_sweep_results.csv`, `registration_threshold_sensitivity.csv`
