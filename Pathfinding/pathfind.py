"""
A* least-cost path between two lon/lat points on Mars, using the USGS
MOLA/HRSC blended DEM. Needs get_mars_patch.py in the same folder.

Usage:
  python pathfind.py --start 137.35 -4.65 --end 137.45 -4.55 --plot
  python pathfind.py --start 137.35 -4.65 --end 137.45 -4.55 --max-slope 15 --margin-km 10 --plot

Install: pip install rasterio numpy scipy matplotlib
"""
import argparse
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from get_mars_patch import get_patch, R


# ---------------------------------------------------------------- geometry
def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a)) / 1000.0


def lonlat_to_rc(lon, lat, info):
    """lon/lat (deg) -> (row, col) pixel in the patch."""
    t = info["transform"]
    x, y = lon, lat
    return int((y - t.f) / t.e), int((x - t.c) / t.a)


def rc_to_lonlat(r, c, info):
    t = info["transform"]
    x, y = t.c + (c + 0.5) * t.a, t.f + (r + 0.5) * t.e
    return x, y


# ---------------------------------------------------------------- cost grid
def build_cost(elev, slope_deg, dx_m, dy_m, max_slope=20.0, slope_weight=4.0,
               rough_weight=0.0, clearance_m=0.0):
    """Return a multiplier grid: 1.0 = flat easy ground, >1 = harder, inf = blocked."""
    cost = 1.0 + slope_weight * (slope_deg / max_slope) ** 2

    if rough_weight > 0:  # local relief in a 3x3 window, in metres
        from scipy.ndimage import maximum_filter, minimum_filter
        relief = maximum_filter(elev, 3) - minimum_filter(elev, 3)
        cost = cost + rough_weight * relief / 100.0

    blocked = (slope_deg > max_slope) | ~np.isfinite(elev) | ~np.isfinite(slope_deg)
    if clearance_m > 0:  # keep away from blocked cells (rover radius / safety margin)
        pix = float(min(dy_m, np.min(dx_m)))
        blocked = distance_transform_edt(~blocked) * pix < clearance_m
    cost = cost.astype(np.float64)
    cost[blocked] = np.inf
    return cost


# ---------------------------------------------------------------- A*
def astar(cost, start, goal, dx_m, dy_m):
    """8-connected A*. Step length uses real metres (dx_m is per row).
    Heuristic = straight-line metres; admissible because cost multiplier >= 1."""
    h, w = cost.shape
    if not np.isfinite(cost[start]) or not np.isfinite(cost[goal]):
        raise ValueError("Start or goal is on blocked terrain (too steep / no data). "
                         "Try other points or a higher --max-slope.")

    def heur(r, c):
        return math.hypot((r - goal[0]) * dy_m, (c - goal[1]) * dx_m[r])

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    g = {start: 0.0}
    prev = {}
    open_heap = [(heur(*start), 0.0, start)]
    closed = set()

    while open_heap:
        f, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1], gc
        closed.add(cur)
        r, c = cur
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in closed or not np.isfinite(cost[nr, nc]):
                continue
            step = math.hypot(dr * dy_m, dc * dx_m[r])
            ng = gc + step * 0.5 * (cost[r, c] + cost[nr, nc])
            if ng < g.get(nxt, math.inf):
                g[nxt] = ng
                prev[nxt] = cur
                heapq.heappush(open_heap, (ng + heur(nr, nc), ng, nxt))
    return None, math.inf


# ---------------------------------------------------------------- helpers
def hillshade(elev, dx_m, dy_m, azimuth=315, altitude=45):
    dzdy = np.gradient(elev, axis=0) / dy_m
    dzdx = np.gradient(elev, axis=1) / dx_m[:, None]
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def path_stats(path, elev, slope, dx_m, dy_m):
    dist = 0.0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        dist += math.hypot((r1 - r0) * dy_m, (c1 - c0) * dx_m[r0])
    return {
        "length_km": dist / 1000,
        "elev_gain_m": float(np.nansum(np.clip(np.diff([elev[p] for p in path]), 0, None))),
        "max_slope_deg": float(np.nanmax([slope[p] for p in path])),
        "mean_slope_deg": float(np.nanmean([slope[p] for p in path])),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser() # parse command line arguments
    ap.add_argument("--start", type=float, nargs=2, required=True, metavar=("LON", "LAT"))
    ap.add_argument("--end", type=float, nargs=2, required=True, metavar=("LON", "LAT"))
    ap.add_argument("--margin-km", type=float, default=5.0,
                    help="extra terrain around the start/end box, lets the path detour")
    ap.add_argument("--max-slope", type=float, default=20.0, help="blocked above this (deg)")
    ap.add_argument("--slope-weight", type=float, default=4.0)
    ap.add_argument("--rough-weight", type=float, default=0.0)
    ap.add_argument("--clearance-m", type=float, default=0.0,
                    help="keep this far from blocked cells (e.g. 200)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="path")
    args = ap.parse_args()

    (lon_a, lat_a), (lon_b, lat_b) = args.start, args.end
    direct = haversine_km(lon_a, lat_a, lon_b, lat_b)
    size_km = direct + 2 * args.margin_km + 1
    print(f"Straight-line distance: {direct:.2f} km; loading {size_km:.1f} km square...")


    lon_c, lat_c = (lon_a + lon_b) / 2, (lat_a + lat_b) / 2
    elev, slope, info = get_patch(lon_c, lat_c, size_km)
    dx_m, dy_m = info["dx_m"], info["dy_m"]
    print(f"Grid: {elev.shape[0]} x {elev.shape[1]} pixels")

    start = lonlat_to_rc(lon_a, lat_a, info)
    goal = lonlat_to_rc(lon_b, lat_b, info)
    for name, (r, c) in (("start", start), ("end", goal)):
        if not (0 <= r < elev.shape[0] and 0 <= c < elev.shape[1]):
            raise SystemExit(f"{name} fell outside the loaded grid; increase --margin-km.")

    cost = build_cost(elev, slope, dx_m, dy_m, args.max_slope, args.slope_weight,
                      args.rough_weight, args.clearance_m)
    print(f"Blocked cells: {np.isinf(cost).mean() * 100:.1f}%")

    path, total = astar(cost, start, goal, dx_m, dy_m)
    if path is None:
        raise SystemExit("No path found.")

    stats = path_stats(path, elev, slope, dx_m, dy_m)
    print(f"Path found: {stats['length_km']:.2f} km "
          f"({stats['length_km'] / direct:.2f}x straight line), "
          f"max slope {stats['max_slope_deg']:.1f} deg, "
          f"mean slope {stats['mean_slope_deg']:.1f} deg, "
          f"total climb {stats['elev_gain_m']:.0f} m")

    ll = np.array([rc_to_lonlat(r, c, info) for r, c in path])
    np.savetxt(f"{args.out}_lonlat.csv", ll, delimiter=",", header="lon,lat", comments="")
    print(f"Saved {args.out}_lonlat.csv ({len(path)} points)")

    if args.plot:
        import matplotlib.pyplot as plt
        pr, pc = zip(*path)
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        ax[0].imshow(hillshade(elev, dx_m, dy_m), cmap="gray")
        ax[0].imshow(elev, cmap="terrain", alpha=0.4)
        ax[0].set_title("Hillshade + elevation")
        ax[1].imshow(slope, cmap="magma", vmax=args.max_slope * 1.5)
        ax[1].contour(np.isinf(cost), levels=[0.5], colors="cyan", linewidths=0.5)
        ax[1].set_title(f"Slope (deg); cyan = blocked (> {args.max_slope} deg)")
        for a in ax:
            a.plot(pc, pr, "r-", lw=1.5)
            a.plot(start[1], start[0], "go"); a.plot(goal[1], goal[0], "bo")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
