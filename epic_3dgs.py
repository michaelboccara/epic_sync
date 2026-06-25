#!/usr/bin/env python3
"""
EPIC DSCOVR → Gaussian Splatting .ply

Fetch natural-color (or enhanced) EPIC frames for a given date from NASA,
or load local images + metadata, then build an anisotropic Gaussian splat PLY.
Uses centroid lat/lon metadata for best-view selection.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

EPIC_BASE = "https://epic.gsfc.nasa.gov"
# Visible Earth disk half-angle in EPIC natural-color images (~22 deg).
DISK_HALF_ANGLE_DEG = 22.0
SIN_THETA_MAX = float(np.sin(np.deg2rad(DISK_HALF_ANGLE_DEG)))


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Uniform points on unit sphere."""
    points = np.zeros((n_points, 3), dtype=np.float32)
    offset = 2.0 / n_points
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = ((i * offset) - 1.0) + (offset / 2.0)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points[i] = [x, y, z]
    return points


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / 0.28209479177387814


def latlon_to_cartesian(lat: float, lon: float) -> np.ndarray:
    """Convert lat/lon (degrees) to unit cartesian vector."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.array([x, y, z], dtype=np.float32)


def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Great-circle angular distance in degrees."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def normals_to_quats(normals: np.ndarray) -> np.ndarray:
    """Unit normals → quaternions aligning local +Z to normal (for flat splats)."""
    n = normals.astype(np.float64)
    axis = np.stack([-n[:, 1], n[:, 0], np.zeros_like(n[:, 0])], axis=1)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(axis_norm > 1e-6, axis / axis_norm, np.array([[1.0, 0.0, 0.0]]))
    cos_a = np.clip(n[:, 2], -1.0, 1.0)
    angle = np.arccos(cos_a)
    half = angle * 0.5
    w = np.cos(half)
    xyz = np.sin(half)[:, None] * axis
    quats = np.concatenate([w[:, None], xyz], axis=1)
    qnorm = np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    return (quats / qnorm).astype(np.float32)


def parse_image_ymd(image_name: str) -> tuple[str, str, str]:
    """Extract year, month, day from an EPIC image filename."""
    ymd = image_name.split("_")[2][:8]
    return ymd[:4], ymd[4:6], ymd[6:8]


def fetch_metadata(date: str, collection: str = "natural") -> list:
    """Fetch EPIC metadata JSON for a calendar date."""
    url = f"{EPIC_BASE}/api/{collection}/date/{date}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def img_archive_url(image_name: str, collection: str, format: str) -> str:
    """Build the archive URL for a full-resolution image."""
    year, month, day = parse_image_ymd(image_name)
    return f"{EPIC_BASE}/archive/{collection}/{year}/{month}/{day}/{format}/{image_name}.{format}"


def download_file(url: str, dest: Path, retries: int = 5) -> None:
    """Download a URL to dest, skipping if already cached."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url) as resp:
                dest.write_bytes(resp.read())
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def load_image_array(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def fetch_day(date: str, cache_dir: Path, collection: str = "natural", format: str = "png") -> tuple[list, dict]:
    """Fetch metadata and images for one day; return (meta_list, image_dict)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"images_{date}.json"

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta_list = json.load(f)
        print(f"Loaded cached metadata from {meta_path}")
    else:
        meta_list = fetch_metadata(date, collection)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2)
        print(f"Fetched metadata for {len(meta_list)} images on {date}")

    img_dir = cache_dir / format
    image_dict = {}
    for entry in meta_list:
        image_name = entry["image"]
        stem = Path(image_name).stem
        img_path = img_dir / f"{stem}.{format}"
        url = img_archive_url(image_name, collection, format)
        print(f"Downloading image {url} to {img_path}")
        download_file(url, img_path)
        image_dict[stem] = load_image_array(img_path)

    print(f"Loaded {len(image_dict)} images for {date}")
    return meta_list, image_dict


def load_offline(meta_path: str, image_paths: list[str]) -> tuple[list, dict]:
    """Load metadata JSON and local image paths."""
    with open(meta_path, encoding="utf-8") as f:
        meta_list = json.load(f)

    image_dict = {}
    for img_path in image_paths:
        path = Path(img_path)
        image_dict[path.stem] = load_image_array(path)
        print(f"Loaded image {img_path}")

    print(f"Loaded {len(image_dict)} images and {len(meta_list)} metadata entries.")
    return meta_list, image_dict


def tangent_basis(lat: float, lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit nadir and east/north tangent axes for a lat/lon on the sphere."""
    nadir = latlon_to_cartesian(lat, lon)
    lon_rad = np.deg2rad(lon)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0], dtype=np.float32)
    north = np.cross(nadir, east)
    north /= np.linalg.norm(north) + 1e-12
    return nadir, east, north


def point_to_disk_pixels(p: np.ndarray, view: dict) -> tuple[float, float] | None:
    """Map a unit sphere point into EPIC image pixel coords via tangent-plane projection."""
    n = view["nadir"]
    dot_pn = float(np.dot(p, n))
    if dot_pn <= 0.0:
        return None

    t = p - dot_pn * n
    sin_theta = float(np.linalg.norm(t))
    if sin_theta > view["sin_theta_max"]:
        return None

    u = view["cx"] + np.dot(t, view["east"]) / view["sin_theta_max"] * view["r_est"]
    v = view["cy"] - np.dot(t, view["north"]) / view["sin_theta_max"] * view["r_est"]

    du = u - view["cx"]
    dv = v - view["cy"]
    if du * du + dv * dv > view["r_est"] * view["r_est"]:
        return None

    if u < 0 or v < 0 or u >= view["w"] or v >= view["h"]:
        return None

    return u, v


def build_views(meta_list: list, image_dict: dict) -> list:
    """Prepare views with nadir vectors and disk sampling parameters."""
    views = []
    for m in meta_list:
        img_name = Path(m.get("image", "")).stem or m.get("identifier", "")
        if img_name not in image_dict:
            continue
        lat = m["centroid_coordinates"]["lat"]
        lon = m["centroid_coordinates"]["lon"]
        nadir, east, north = tangent_basis(lat, lon)
        img = image_dict[img_name]
        views.append({
            "img": img,
            "nadir": nadir,
            "east": east,
            "north": north,
            "name": img_name,
            "h": img.shape[0],
            "w": img.shape[1],
            "cx": img.shape[1] / 2,
            "cy": img.shape[0] / 2,
            "r_est": min(img.shape[:2]) * 0.48,
            "sin_theta_max": SIN_THETA_MAX,
        })
    return views


def sample_colors(points: np.ndarray, views: list) -> np.ndarray:
    """Sample RGB colors from the best EPIC view per sphere point."""
    colors = np.full((len(points), 3), [0.10, 0.25, 0.50], dtype=np.float32)

    print("Sampling colors from best EPIC view per Gaussian...")
    for i, p in enumerate(points):
        best_dist = 180.0
        best_view = None
        best_u, best_v = 0.0, 0.0

        for v in views:
            dist = angular_distance(p, v["nadir"])
            if dist >= best_dist or dist >= DISK_HALF_ANGLE_DEG:
                continue
            uv = point_to_disk_pixels(p, v)
            if uv is None:
                continue
            best_dist = dist
            best_view = v
            best_u, best_v = uv

        if best_view is not None:
            coords = np.array([[best_v], [best_u]])
            for ch in range(3):
                try:
                    colors[i, ch] = map_coordinates(
                        best_view["img"][:, :, ch], coords, order=1, mode="nearest"
                    )[0]
                except (IndexError, ValueError):
                    pass

    mean_rgb = colors.mean(axis=0)
    near_black = (colors.max(axis=1) < 0.05).mean()
    print(
        f"Color stats — mean RGB: [{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}], "
        f"near-black: {near_black:.1%}"
    )
    return colors


def write_splat_ply(output: str, points: np.ndarray, colors: np.ndarray, args) -> None:
    """Write a 3D Gaussian splat binary PLY."""
    print(f"Writing splat ply to {output}")
    n = len(points)
    sh_dc = rgb_to_sh(colors)
    normals = points.copy()

    positions = points * args.radius
    opacities = np.full(n, 2.197, dtype=np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    # 3DGS PLY stores log-scale; viewers decode with exp().
    scales[:, 0:2] = np.log(args.base_scale * 1.15)
    scales[:, 2] = np.log(args.base_scale * 0.22)
    rots = normals_to_quats(normals)

    data = np.empty((n, 14), dtype=np.float32)
    data[:, 0:3] = positions
    data[:, 3:6] = sh_dc
    data[:, 6] = opacities
    data[:, 7:10] = scales
    data[:, 10:14] = rots

    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    with open(output, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)

    print(f"Saved {output} ({data.nbytes / (1024 * 1024):.2f} MB)")


def build_splat_ply(meta_list: list, image_dict: dict, args) -> None:
    print(f"Building splat ply for {args.output}")
    views = build_views(meta_list, image_dict)
    if not views:
        raise RuntimeError("No views matched metadata to images.")

    print(f"Prepared {len(views)} views for texturing.")
    points = fibonacci_sphere(args.n_gaussians)
    colors = sample_colors(points, views)
    write_splat_ply(args.output, points, colors, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Gaussian splat PLY from NASA EPIC Earth imagery."
    )
    parser.add_argument("--date", type=str, help="Calendar date (YYYY-MM-DD) to fetch from EPIC")
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory for cached metadata and images (default: ./epic_cache/{date})",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="EPIC image format: jpg or png (default: png)",
    )
    parser.add_argument(
        "--collection",
        choices=["natural", "enhanced"],
        default="natural",
        help="EPIC image collection (default: natural)",
    )
    parser.add_argument("--images", nargs="+", help="Paths to local image images (offline mode)")
    parser.add_argument("--metadata", type=str, help="Path to JSON metadata list (offline mode)")
    parser.add_argument("--output", help="Output .ply path")
    parser.add_argument("--n_gaussians", type=int, default=150000)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--base_scale", type=float, default=0.007)
    args = parser.parse_args()

    online = args.date is not None
    offline = args.images is not None or args.metadata is not None

    if online and offline:
        parser.error("Use either --date (online) or --images/--metadata (offline), not both.")
    if not online and not (args.images and args.metadata):
        parser.error("Provide --date for online fetch, or both --images and --metadata for offline.")

    if online and args.output is None:
        args.output = f"earth_{args.date}.ply"
    elif args.output is None:
        args.output = "earth_epic_gs.ply"

    if online and args.cache_dir is None:
        args.cache_dir = f"./epic_cache/{args.date}"

    return args


def main():
    args = parse_args()

    if args.date:
        meta_list, image_dict = fetch_day(args.date, Path(args.cache_dir), args.collection, args.format)
    else:
        meta_list, image_dict = load_offline(args.metadata, args.images)

    build_splat_ply(meta_list, image_dict, args)


if __name__ == "__main__":
    main()
