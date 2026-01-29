# Technical Accuracy Report — Python and C++ (ICP + RANSAC 3D Registration)

## What was measured

Three pipelines were evaluated under several synthetic configurations:

- **A0**: nearest-neighbor correspondences → RANSAC initialization → ICP refinement  
- **B**: ICP refinement from a good initial pose  
- **C**: RANSAC initialization from correspondence candidates → ICP refinement  

Each configuration reports:
- success rate per pipeline
- median rotation error (radians)
- median inlier count for pipeline C

---

## Synthetic configurations

Each configuration label encodes the synthetic conditions:
- noise level
- outlier ratio
- overlap ratio (when < 100%)

The experiment settings used by the C++ runner here were:
- `trials = 3`
- `seed0 = 21`

(The Python table below corresponds to the latest stored run under the same set of configuration labels.)

---

## Python accuracy (latest stored results)

| config                                           |   trials |   A0_success_rate |   B_success_rate |   C_success_rate |   A0_med_rot_rad |   B_med_rot_rad |   C_med_rot_rad |   C_med_inliers |
|:-------------------------------------------------|---------:|------------------:|-----------------:|-----------------:|-----------------:|----------------:|----------------:|----------------:|
| hard: noise 0.02, outliers 40%, overlap 100%     |        5 |                 0 |                1 |                1 |          3.09267 |     0.00130624  |     0.00130624  |             121 |
| moderate: noise 0.01, outliers 20%, overlap 100% |        5 |                 0 |                1 |                1 |          2.70631 |     0.000956155 |     0.000960892 |             173 |
| partial: noise 0.01, outliers 20%, overlap 60%   |        5 |                 0 |                1 |                1 |          2.89695 |     0.000988347 |     0.00103254  |             210 |

---

## C++ accuracy (Armadillo, C++20)

| Config | Trials | A0 success | B success | C success | A0 med rot | B med rot | C med rot | C med inliers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard: noise 0.02, outliers 40%, overlap 100% | 3 | 0.000000 | 1.000000 | 1.000000 | 2.852261 | 0.001725 | 0.001797 | 120.000000 |
| moderate: noise 0.01, outliers 20%, overlap 100% | 3 | 0.000000 | 1.000000 | 1.000000 | 2.678284 | 0.000581 | 0.000581 | 174.000000 |
| partial: noise 0.01, outliers 20%, overlap 60% | 3 | 0.000000 | 1.000000 | 1.000000 | 2.858140 | 0.000847 | 0.000847 | 210.000000 |
