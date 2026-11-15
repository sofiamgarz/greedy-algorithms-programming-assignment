"""
Tests and experiments for k_center.py: correctness checks, timings,
approximation ratios, the figure of the report and the tight family
used in the optional part.

Run with:  python experiments.py
"""

import math
import os
import random
import statistics
import time
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from k_center import (greedy_k_center, optimal_k_center, covering_radius,
                      _farthest_first)

TOL = 1e-9
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# --------------------------------------------------------------------------
# instance generators
# --------------------------------------------------------------------------

def random_int_points(n, rng, size=20):
    """n distinct points with integer coordinates in [0, size]^2."""
    cells = [(x, y) for x in range(size + 1) for y in range(size + 1)]
    return rng.sample(cells, n)


def random_float_points(n, rng, side=100.0):
    """n distinct points drawn uniformly from a square of the given side."""
    pts = set()
    while len(pts) < n:
        pts.add((rng.uniform(0, side), rng.uniform(0, side)))
    pts = list(pts)
    rng.shuffle(pts)
    return pts


def grid_points(rows, cols):
    return [(x, y) for x in range(cols) for y in range(rows)]


def cluster_points(cluster_centers, per_cluster, spread, rng):
    """Small groups of points around a few far apart locations."""
    pts = set()
    for (cx, cy) in cluster_centers:
        group = set()
        while len(group) < per_cluster:
            group.add((cx + rng.randint(-spread, spread),
                       cy + rng.randint(-spread, spread)))
        pts |= group
    return sorted(pts)


def tight_instance(k, gap=10):
    """
    k blocks of three consecutive points on a line, blocks far from each other.
    Greedy starts at the left end of the first block, so it gets radius 2,
    while the middle points give radius 1.
    """
    return [(gap * j + i, 0) for j in range(k) for i in range(3)]


# --------------------------------------------------------------------------
# correctness checks
# --------------------------------------------------------------------------

def naive_farthest_first(points, k):
    """
    Reference greedy that recomputes every distance from scratch at each step.
    Slow, but it does not share any code with greedy_k_center.
    """
    centers = [min(points)]
    while len(centers) < k:
        best, best_d = None, -1.0
        for p in sorted(points):
            if p in centers:
                continue
            d = min(math.dist(p, c) for c in centers)
            if d > best_d + TOL:
                best, best_d = p, d
        centers.append(best)
    return sorted(centers)


def brute_force_reference(points, k):
    """Smallest radius (and first subset reaching it) using covering_radius."""
    best_r, best_set = math.inf, None
    for cand in combinations(sorted(points), k):
        r = covering_radius(points, cand)
        if r < best_r - TOL:
            best_r, best_set = r, list(cand)
    return best_r, best_set


def check_instance(points, k, exact_ties=True):
    """
    Run both algorithms on one instance and collect every problem found.
    Returns (greedy radius, optimal radius, list of error messages).
    """
    errors = []
    original = list(points)
    g = greedy_k_center(points, k)
    o = optimal_k_center(points, k)

    if points != original:
        errors.append("input list was modified")

    for name, sol in (("greedy", g), ("optimal", o)):
        if not isinstance(sol, list) or len(sol) != k or len(set(sol)) != k:
            errors.append(name + ": does not return k distinct centers")
        if not set(sol) <= set(points):
            errors.append(name + ": center that is not an input point")
        if sol != sorted(sol):
            errors.append(name + ": result is not sorted")

    rg = covering_radius(points, g)
    ro = covering_radius(points, o)
    ref_r, ref_set = brute_force_reference(points, k)

    if abs(ro - ref_r) > TOL:
        errors.append("optimal: radius %.6f but reference %.6f" % (ro, ref_r))
    if rg < ro - TOL:
        errors.append("greedy radius below the optimum")
    if rg > 2 * ro + TOL:
        errors.append("bound Rg <= 2R* violated (%.6f > 2 * %.6f)" % (rg, ro))
    if k == len(points) and (rg > TOL or ro > TOL):
        errors.append("k = n should give radius 0")

    # tie-breaking can only be compared exactly on integer coordinates
    if exact_ties:
        if o != ref_set:
            errors.append("optimal: not the lexicographically smallest set")
        if g != naive_farthest_first(points, k):
            errors.append("greedy: differs from the naive farthest-first")

    return rg, ro, errors


def run_correctness_checks():
    print("=" * 70)
    print("CORRECTNESS CHECKS")
    print("=" * 70)

    # 1) checks given in the statement
    P = [(0, 0), (2, 0), (3, 0), (5, 0), (6, 0), (8, 0)]
    statement = [
        ("greedy(P, 2)", greedy_k_center(P, 2) == [(0, 0), (8, 0)]),
        ("optimal(P, 2)", optimal_k_center(P, 2) == [(2, 0), (6, 0)]),
        ("radius greedy = 3", abs(covering_radius(P, [(0, 0), (8, 0)]) - 3) < 1e-9),
        ("radius optimal = 2", abs(covering_radius(P, [(2, 0), (6, 0)]) - 2) < 1e-9),
    ]
    for name, ok in statement:
        print("  %-22s %s" % (name, "ok" if ok else "FAILED"))
    failed_statement = sum(1 for _, ok in statement if not ok)

    # 2) families of small instances: (points, k, exact tie comparison)
    rng = random.Random(2026)
    families = {}

    base_sets = [
        [(0, 0), (2, 0), (3, 0), (5, 0), (6, 0), (8, 0)],
        grid_points(3, 3),
        random_int_points(9, rng),
        random_int_points(10, rng),
        [(4, 1)],
    ]
    families["k = 1"] = [(pts, 1, True) for pts in base_sets]
    families["k = n"] = [(pts, len(pts), True) for pts in base_sets]

    lines = [
        [(i, 0) for i in range(9)],                       # horizontal
        [(0, i) for i in range(9)],                       # vertical
        [(i, i) for i in range(8)],                       # diagonal
        [(2 * i, -3 * i) for i in range(8)],
        [(x, 0) for x in (0, 1, 2, 4, 7, 11, 16, 22)],    # uneven gaps
        [(x, 2 * x + 1) for x in (-5, -4, 0, 1, 3, 8, 9)],
    ]
    families["collinear"] = [(pts, k, True)
                             for pts in lines for k in range(1, len(pts) + 1)]

    grids = [grid_points(2, 2), grid_points(3, 3), grid_points(3, 4),
             grid_points(4, 4), grid_points(2, 6)]
    families["grids"] = [(pts, k, True)
                         for pts in grids for k in range(1, min(len(pts), 5) + 1)]

    sep = []
    for _ in range(10):
        three = cluster_points([(0, 0), (100, 0), (50, 90)], 4, 3, rng)
        four = cluster_points([(0, 0), (100, 0), (0, 100), (100, 100)], 3, 3, rng)
        sep += [(three, k, True) for k in (2, 3, 4)]
        sep += [(four, k, True) for k in (3, 4, 5)]
    families["separated clusters"] = sep

    rand_int = []
    for _ in range(300):
        n = rng.randint(2, 10)
        rand_int.append((random_int_points(n, rng), rng.randint(1, n), True))
    families["random (integer)"] = rand_int

    rand_float = []
    for _ in range(300):
        n = rng.randint(2, 10)
        rand_float.append((random_float_points(n, rng), rng.randint(1, n), False))
    families["random (float)"] = rand_float

    print()
    print("  %-20s %9s %8s %10s" % ("family", "instances", "failed", "max Rg/R*"))
    total, total_failed = 0, failed_statement
    for name, cases in families.items():
        failed, worst = 0, 0.0
        for pts, k, exact in cases:
            rg, ro, errors = check_instance(pts, k, exact)
            if errors:
                failed += 1
                print("    FAILED", name, "n=%d k=%d:" % (len(pts), k), errors)
            if ro > TOL:
                worst = max(worst, rg / ro)
        total += len(cases)
        total_failed += failed
        print("  %-20s %9d %8d %10.3f" % (name, len(cases), failed, worst))

    print()
    print("  total: %d instances, %d failures" % (total, total_failed))
    return total_failed


# --------------------------------------------------------------------------
# timing experiments
# --------------------------------------------------------------------------

def median_time(func, points, k, repeats=5):
    """Median wall-clock time (seconds) of func(points, k) over some runs."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(points, k)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_timing_experiments():
    print()
    print("=" * 70)
    print("TIMINGS (median of 5 runs)")
    print("=" * 70)
    rng = random.Random(7)

    # sizes where the exhaustive search is still reasonable
    small = [(10, 3), (20, 3), (30, 3), (40, 3), (50, 3),
             (10, 5), (15, 5), (20, 5), (25, 5),
             (8, 4), (12, 6), (16, 8), (18, 9)]
    print("  %5s %3s %12s %14s %14s" % ("n", "k", "C(n,k)", "greedy (ms)", "exact (s)"))
    last = None
    for n, k in small:
        pts = random_float_points(n, rng)
        tg = median_time(greedy_k_center, pts, k)
        to = median_time(optimal_k_center, pts, k)
        print("  %5d %3d %12d %14.4f %14.4f" % (n, k, math.comb(n, k), tg * 1e3, to))
        last = (n, k, to)

    # cost of one distance evaluation inside the exhaustive search, used to
    # estimate the sizes that cannot be run
    n, k, to = last
    unit = to / (math.comb(n, k) * n * k)
    print()
    print("  approx. cost per distance evaluation in exact: %.1f ns" % (unit * 1e9))
    for n, k in [(50, 5), (100, 5), (40, 20), (100, 10)]:
        est = unit * math.comb(n, k) * n * k
        print("  estimated exact time for n=%d, k=%d: %.3g s  (%.3g years)"
              % (n, k, est, est / (3600 * 24 * 365)))

    # greedy alone on much larger inputs
    print()
    print("  %8s %4s %14s" % ("n", "k", "greedy (ms)"))
    for n, k in [(1000, 5), (10000, 5), (100000, 5), (10000, 10), (10000, 20), (10000, 40)]:
        pts = random_float_points(n, rng)
        tg = median_time(greedy_k_center, pts, k)
        print("  %8d %4d %14.2f" % (n, k, tg * 1e3))


# --------------------------------------------------------------------------
# approximation ratios
# --------------------------------------------------------------------------

def ratio_row(name, points, k):
    g = greedy_k_center(points, k)
    o = optimal_k_center(points, k)
    rg, ro = covering_radius(points, g), covering_radius(points, o)
    print("  %-26s %3d %3d %8.3f %8.3f %7.3f" % (name, len(points), k, rg, ro, rg / ro))


def run_ratio_experiments():
    print()
    print("=" * 70)
    print("APPROXIMATION RATIOS")
    print("=" * 70)
    rng = random.Random(11)

    print("  %-26s %3s %3s %8s %8s %7s" % ("instance", "n", "k", "Rg", "R*", "ratio"))
    ratio_row("statement example", [(0, 0), (2, 0), (3, 0), (5, 0), (6, 0), (8, 0)], 2)
    ratio_row("line 0..4", [(i, 0) for i in range(5)], 2)
    ratio_row("line 0..8", [(i, 0) for i in range(9)], 3)
    ratio_row("grid 4x4", grid_points(4, 4), 3)
    ratio_row("grid 5x5", grid_points(5, 5), 4)
    ratio_row("3 separated clusters",
              cluster_points([(0, 0), (100, 0), (50, 90)], 5, 4, rng), 3)
    ratio_row("3 clusters, k = 2",
              cluster_points([(0, 0), (100, 0), (50, 90)], 5, 4, rng), 2)
    ratio_row("random integer (n=14)", random_int_points(14, rng), 3)
    ratio_row("random float (n=14)", random_float_points(14, rng), 4)
    ratio_row("random float (n=18)", random_float_points(18, rng), 5)

    # distribution of the ratio over many random instances
    print()
    print("  random float instances, n = 14, 300 instances per k")
    print("  %3s %8s %8s %8s %12s" % ("k", "mean", "median", "max", "greedy opt."))
    for k in (2, 3, 4, 5):
        ratios = []
        for _ in range(300):
            pts = random_float_points(14, rng)
            rg = covering_radius(pts, greedy_k_center(pts, k))
            ro = covering_radius(pts, optimal_k_center(pts, k))
            ratios.append(rg / ro)
        optimal_share = sum(1 for r in ratios if r < 1 + TOL) / len(ratios)
        print("  %3d %8.3f %8.3f %8.3f %11.1f%%" % (
            k, statistics.mean(ratios), statistics.median(ratios),
            max(ratios), 100 * optimal_share))


def run_tight_family():
    print()
    print("=" * 70)
    print("TIGHT FAMILY (k blocks of 3 collinear points)")
    print("=" * 70)
    print("  %3s %4s %6s %6s %7s" % ("k", "n", "Rg", "R*", "ratio"))
    for k in range(1, 7):
        pts = tight_instance(k)
        rg = covering_radius(pts, greedy_k_center(pts, k))
        ro = covering_radius(pts, optimal_k_center(pts, k))
        print("  %3d %4d %6.2f %6.2f %7.3f" % (k, len(pts), rg, ro, rg / ro))


# --------------------------------------------------------------------------
# figure and effect of the first center
# --------------------------------------------------------------------------

def figure_instance():
    """Three loose groups of points of different sizes (60 points in total)."""
    rng = random.Random(3)
    pts = set()
    for (cx, cy), size, sigma in [((2.5, 2.5), 24, 1.1), ((8.0, 3.5), 16, 0.9),
                                  ((5.0, 8.0), 20, 1.3)]:
        group = set()
        while len(group) < size:
            group.add((rng.gauss(cx, sigma), rng.gauss(cy, sigma)))
        pts |= group
    pts = list(pts)
    rng.shuffle(pts)
    return pts


def make_figure(points, k, filename):
    centers = greedy_k_center(points, k)
    radius = covering_radius(points, centers)

    # nearest-center assignment (index of the closest center for every point)
    assign = [min(range(k), key=lambda j: math.dist(p, centers[j])) for p in points]
    # the point that defines the covering radius
    worst = max(points, key=lambda p: min(math.dist(p, c) for c in centers))
    worst_center = min(centers, key=lambda c: math.dist(worst, c))

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    for j, c in enumerate(centers):
        ax.add_patch(Circle(c, radius, fill=True, facecolor=colors[j], alpha=0.07,
                            edgecolor="none", zorder=1))
        ax.add_patch(Circle(c, radius, fill=False, edgecolor=colors[j],
                            linestyle="--", linewidth=1.1, zorder=2))
        cluster = [p for p, a in zip(points, assign) if a == j and p != c]
        for p in cluster:
            ax.plot([p[0], c[0]], [p[1], c[1]], color=colors[j],
                    linewidth=0.6, alpha=0.45, zorder=2)
        ax.scatter([p[0] for p in cluster], [p[1] for p in cluster], s=30,
                   marker=markers[j], color=colors[j], edgecolor="white",
                   linewidth=0.6, zorder=3, label="assigned to center %d" % (j + 1))

    ax.scatter([c[0] for c in centers], [c[1] for c in centers], s=230, marker="*",
               color="#0b0b0b", edgecolor="white", linewidth=0.9, zorder=5,
               label="greedy centers")

    # segment from the worst covered point to its center, i.e. the radius
    ax.plot([worst[0], worst_center[0]], [worst[1], worst_center[1]],
            color="#0b0b0b", linewidth=1.6, zorder=4)
    ax.scatter([worst[0]], [worst[1]], s=110, facecolor="none", edgecolor="#0b0b0b",
               linewidth=1.4, zorder=4, label="farthest point")
    ax.annotate(r"$R_g = %.2f$" % radius, worst, textcoords="offset points",
                xytext=(11, -3), fontsize=10)

    # keep the view on the data, the circles are clipped by the axes
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Farthest-first traversal, n = %d, k = %d" % (len(points), k),
                 fontsize=11)
    ax.grid(True, color="#e6e6e3", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False, fontsize=9)

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path, centers, radius


def run_first_center_experiment(points, k):
    """Radius obtained when every input point is tried as the first center."""
    print()
    print("=" * 70)
    print("EFFECT OF THE FIRST CENTER (figure instance, k = %d)" % k)
    print("=" * 70)
    radii = []
    for start in range(len(points)):
        centers = _farthest_first(points, k, start)
        radii.append(covering_radius(points, centers))
    convention = covering_radius(points, greedy_k_center(points, k))
    opt = covering_radius(points, optimal_k_center(points, k))
    print("  radius with our convention : %.3f" % convention)
    print("  best / worst over all starts: %.3f / %.3f" % (min(radii), max(radii)))
    print("  different radii obtained    : %d" % len(set(round(r, 9) for r in radii)))
    print("  optimal radius              : %.3f" % opt)

    line = [(i, 0) for i in range(5)]
    print("  line 0..4, k = 2:")
    for start in range(len(line)):
        centers = _farthest_first(line, 2, start)
        print("    first center %s -> centers %s, radius %.0f"
              % (line[start], sorted(centers), covering_radius(line, centers)))


if __name__ == "__main__":
    failures = run_correctness_checks()
    run_timing_experiments()
    run_ratio_experiments()
    run_tight_family()

    pts = figure_instance()
    path, centers, radius = make_figure(pts, 3, "greedy_k_center.png")
    print()
    print("figure saved in", path)
    print("centers:", [(round(x, 2), round(y, 2)) for x, y in centers],
          "radius: %.3f" % radius)
    run_first_center_experiment(pts, 3)

    print()
    print("correctness checks failed:", failures)
