# Greedy Algorithms: Geometric k-Center

Programming assignment for the Computational Complexity course.

Author: Sofía Muñiz Garzón

## Problem

Given a set `P` of `n` distinct points in the plane and an integer `k`, choose `k` of
the points as centers so that the largest distance from a point to its nearest center
(the covering radius) is as small as possible. Distances are Euclidean and the centers
must be input points (discrete Euclidean k-center).

## Files

- `k_center.py`: the three functions required by the assignment.
  - `greedy_k_center(points, k)`: farthest-first traversal, a 2-approximation.
  - `optimal_k_center(points, k)`: exhaustive search over all k-subsets, only for
    small inputs.
  - `covering_radius(points, centers)`: value of the objective.
- `experiments.py`: correctness checks, timings, approximation ratios, the figure
  and the family of instances where the ratio is exactly 2.

## Usage

`k_center.py` only needs the Python standard library (Python 3.8 or later).

```python
from k_center import greedy_k_center, optimal_k_center, covering_radius

P = [(0, 0), (2, 0), (3, 0), (5, 0), (6, 0), (8, 0)]

greedy_k_center(P, 2)                    # [(0, 0), (8, 0)]
optimal_k_center(P, 2)                   # [(2, 0), (6, 0)]
covering_radius(P, [(0, 0), (8, 0)])     # 3.0
covering_radius(P, [(2, 0), (6, 0)])     # 2.0
```

Both selection functions return the centers sorted in lexicographic order. The greedy
algorithm starts from the lexicographically smallest point and breaks ties in the same
way, so the output is deterministic.

## Experiments

```
pip install matplotlib
python experiments.py
```

The script prints the results of the checks and the tables, and saves the figure in
`figures/greedy_k_center.png`. It takes about half a minute.

## Complexity

| Function | Time | Auxiliary space |
|---|---|---|
| `greedy_k_center` | Θ(nk) | Θ(n) |
| `optimal_k_center` | Θ(C(n,k) · nk) | Θ(n + k) |
| `covering_radius` | Θ(n · \|C\|) | Θ(1) |

## References

- T. F. Gonzalez (1985). Clustering to minimize the maximum intercluster distance.
  *Theoretical Computer Science*, 38, 293–306.
- D. S. Hochbaum and D. B. Shmoys (1985). A best possible heuristic for the k-center
  problem. *Mathematics of Operations Research*, 10(2), 180–184.
