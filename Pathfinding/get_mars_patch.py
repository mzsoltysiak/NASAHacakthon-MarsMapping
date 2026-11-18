import argparse
import numpy as np
import rasterio
from rasterio.windows import from_bounds

R = 3396190.0  # Mars radius in metres (from the dataset metadata)
URL = (
    "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/"
    "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
)


def get_patch(lon, lat, size_km=10.0, url=URL):
    """Return (elev_m, slope_deg, info) for a size_km x size_km square
    centred on lon/lat (degrees, planetocentric, east positive)."""
    if abs(lat) > 85:
        raise ValueError("Too close to a pole: cylindrical projection is badly distorted.")
    lon = ((lon + 180) % 360) - 180          # force -180..180
    half = size_km * 1000.0 / 2.0

    # Only read what we need from the file (HTTP range requests).
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_UNSAFESSL="YES",
    ):
        with rasterio.open(url) as src:
            # The file's coordinates are in degrees

            # Ground distance east-west per unit of longitude shrinks by cos(lat),
            # so a true 10 km square is wider than 10 km in map coordinates.
            dlat_deg = np.degrees(half / R)
            dlon_deg = np.degrees(half / (R * np.cos(np.radians(lat))))

            lat0, lat1 = lat - dlat_deg, lat + dlat_deg
            lon0, lon1 = lon - dlon_deg, lon + dlon_deg

            
            x0, x1, y0, y1 = lon0, lon1, lat0, lat1

            win = from_bounds(x0, y0, x1, y1, src.transform)
            win = win.round_offsets().round_lengths()
            elev = src.read(1, window=win, boundless=False).astype("float32")

            if src.nodata is not None:
                elev[elev == src.nodata] = np.nan
            elev = elev * (src.scales[0] or 1.0) + (src.offsets[0] or 0.0)

            # Per-row latitude, to get true ground size of each pixel.
            t = src.window_transform(win)
            ys = t.f + (np.arange(elev.shape[0]) + 0.5) * t.e
            xres, yres = abs(t.a), abs(t.e)

            print("Degrees")
            row_lat = ys
            dy = np.radians(yres) * R
            dx = np.radians(xres) * R * np.cos(np.radians(row_lat))

    # Slope from elevation gradient, using true ground spacing.
    dz_dy = np.gradient(elev, axis=0) / dy
    dz_dx = np.gradient(elev, axis=1) / dx[:, None]
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    info = {
        "shape": elev.shape,
        "pixel_ns_m": float(dy),
        "pixel_ew_m": float(dx.mean()),
        "extent_ns_km": elev.shape[0] * float(dy) / 1000,
        "extent_ew_km": elev.shape[1] * float(dx.mean()) / 1000,
        # extra georeferencing, used by pathfind.py
        "transform": t,
        "dx_m": dx,   # metres per pixel E-W, one value per row
        "dy_m": dy,   # metres per pixel N-S
    }
    return elev, slope_deg, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, required=True, help="centre longitude, deg east")
    ap.add_argument("--lat", type=float, required=True, help="centre latitude, deg")
    ap.add_argument("--size-km", type=float, default=10.0)
    ap.add_argument("--out", default="patch")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    elev, slope, info = get_patch(args.lon, args.lat, args.size_km)
    print({k: v for k, v in info.items() if k not in ("transform", "dx_m", "dy_m")})
    print(f"elevation: min {np.nanmin(elev):.1f} m, max {np.nanmax(elev):.1f} m")
    print(f"slope:     mean {np.nanmean(slope):.2f} deg, max {np.nanmax(slope):.2f} deg")

    np.save(f"{args.out}_elev.npy", elev)
    np.save(f"{args.out}_slope.npy", slope)
    print(f"saved {args.out}_elev.npy and {args.out}_slope.npy")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        im0 = ax[0].imshow(elev, cmap="terrain")
        ax[0].set_title("Elevation (m)"); fig.colorbar(im0, ax=ax[0], shrink=0.8)
        im1 = ax[1].imshow(slope, cmap="magma")
        ax[1].set_title("Slope (deg)"); fig.colorbar(im1, ax=ax[1], shrink=0.8)
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
