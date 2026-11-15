"""
Discrete Euclidean k-center in the plane.

    greedy_k_center   farthest-first traversal (Gonzalez, 1985)
    optimal_k_center  exhaustive search, only for small inputs
    covering_radius   value of the objective R(P, C)

Centers are always taken from the input points.
"""

import math
from itertools import combinations


def _sq_dist(p, q):
    """Squared Euclidean distance between p and q."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


# Both selection functions compare squared distances. The ordering is the same
# as with the real distances, but with integer coordinates the values are exact
# integers, so ties are detected without any rounding issue.


def _farthest_first(points, k, start):
    """
    Farthest-first traversal starting from points[start].

    Returns the k centers in the order in which they were selected.
    """
    n = len(points)
    is_center = [False] * n
    is_center[start] = True
    centers = [points[start]]

    # nearest[i] = squared distance from points[i] to its closest center so far
    nearest = [_sq_dist(p, points[start]) for p in points]

    for _ in range(k - 1):
        # greedy choice: the point that is worst covered right now,
        # ties go to the lexicographically smallest one
        far = -1
        for i in range(n):
            if is_center[i]:
                continue
            if (far == -1 or nearest[i] > nearest[far]
                    or (nearest[i] == nearest[far] and points[i] < points[far])):
                far = i

        is_center[far] = True
        centers.append(points[far])

        # only the new center can improve a point, so one pass is enough
        new = points[far]
        for i in range(n):
            d = _sq_dist(points[i], new)
            if d < nearest[i]:
                nearest[i] = d

    return centers


def greedy_k_center(points, k):
    """Return k selected centers as a list of (x, y) tuples."""
    # first center: lexicographically smallest input point
    start = 0
    for i in range(1, len(points)):
        if points[i] < points[start]:
            start = i
    return sorted(_farthest_first(points, k, start))


def optimal_k_center(points, k):
    """Return an optimal set of k centers, for small instances."""
    pts = sorted(points)
    best = None
    best_radius = math.inf

    # combinations() of a sorted list come out in lexicographic order, so by
    # keeping only strict improvements the first optimal subset found is also
    # the lexicographically smallest one
    for cand in combinations(pts, k):
        radius = 0          # squared covering radius of this candidate
        for p in pts:
            d = min(_sq_dist(p, c) for c in cand)
            if d > radius:
                radius = d
        if radius < best_radius:
            best_radius = radius
            best = cand

    return list(best)


def covering_radius(points, centers):
    """Return the maximum distance to the nearest chosen center."""
    radius = 0.0
    for p in points:
        closest = min(math.dist(p, c) for c in centers)
        radius = max(radius, closest)
    return radius


if __name__ == "__main__":
    # example from the assignment statement
    P = [(0, 0), (2, 0), (3, 0), (5, 0), (6, 0), (8, 0)]
    assert greedy_k_center(P, 2) == [(0, 0), (8, 0)]
    assert optimal_k_center(P, 2) == [(2, 0), (6, 0)]
    assert abs(covering_radius(P, [(0, 0), (8, 0)]) - 3) < 1e-9
    assert abs(covering_radius(P, [(2, 0), (6, 0)]) - 2) < 1e-9
    print("greedy :", greedy_k_center(P, 2), covering_radius(P, greedy_k_center(P, 2)))
    print("optimal:", optimal_k_center(P, 2), covering_radius(P, optimal_k_center(P, 2)))
