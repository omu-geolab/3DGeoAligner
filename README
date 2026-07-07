# 3D Geo Aligner

**A low-cost pipeline for wide-area 3D mapping using LiDAR-equipped mobile devices and open data.**

This repository provides the reference implementation of the method described in:

> Ueda, R., Yoshida, D. *A Pipeline for Low-Cost Wide-Area 3D Mapping Using LiDAR-Equipped Mobile Devices and Open Data.* (ISPRS, 2026)

The pipeline integrates multiple locally captured point clouds (e.g. from an iPhone/iPad LiDAR scanner) into a single, georeferenced 3D map, using only consumer mobile devices and publicly available open geospatial data — no survey-grade equipment (MMS/TLS/GNSS base stations) required.

The core idea is to **decompose 3D registration and georeferencing into independent horizontal and vertical steps**, which improves robustness against the noise and drift inherent in mobile LiDAR scanning while keeping computation lightweight.

Validation in Tondabayashi City, Japan achieved a horizontal RMSE of 1.75 m and a vertical RMSE of 0.10 m, meeting the accuracy requirements for 1:2,500-scale disaster prevention base maps.

## Pipeline Overview

| Stage | Script | Description |
|---|---|---|
| **Registration** | `scripts/registration.py` | Aligns a source point cloud to a target (adjacent) point cloud. Extracts the mutually overlapping region, separates ground / non-ground points via Cloth Simulation Filter (CSF), then performs horizontal registration (2D rasterization → ORB feature matching → 2D ICP) followed by vertical registration (planar least-squares fit on ground points). |
| **Georeferencing** | `scripts/georeference.py` | Assigns absolute coordinates to the registered/integrated point cloud by matching non-ground wall features against open road-edge vector data (horizontal) and fitting ground points to a public Digital Elevation Model, or DEM (vertical). |

Both scripts follow the same underlying strategy from the paper: separate ground/non-ground points with CSF, then solve the horizontal and vertical problems independently.

## Repository Structure

```
.
├── docker-compose.yaml
├── Dockerfile
├── scripts/
│   ├── registration.py       # Point cloud ↔ point cloud registration
│   └── georeference.py       # Point cloud ↔ open data georeferencing
└── data/
    ├── scaniverse_tondabayashi_01.las
    ├── scaniverse_tondabayashi_02.las
    ├── scaniverse_tondabayashi_03.las
    ├── scaniverse_tondabayashi_04.las
    └── scaniverse_tondabayashi_05.las
```

## Demo Data

Five sample scans captured with an iPad Pro (LiDAR) using the **Scaniverse** app are provided in `data/` for trying out the pipeline end to end. The scans were collected along adjacent, overlapping routes in the Jinaimachi district, Tondabayashi City, Osaka, Japan, and connect in a **T-shape** layout:

```
        Scan 5
          |
Scan 1 — Scan 2 — Scan 3 — Scan 4
```

Adjacent, overlapping pairs suitable for registration are:

- `01` ↔ `02`
- `02` ↔ `03`
- `03` ↔ `04`
- `02` ↔ `05`
- `03` ↔ `05`

These pairs can be chained to reconstruct the full T-shaped area (e.g. register `01`→`02`, then use the result as the new target for `03`, and so on).

## Requirements

The pipeline is built on the following open-source components:

- [Open3D](http://www.open3d.org/) — point cloud I/O, filtering, ICP
- [laspy](https://laspy.readthedocs.io/) — LAS/LAZ file I/O
- [OpenCV](https://opencv.org/) — ORB feature matching, image processing
- [CSF](https://github.com/jianboqi/CSF) — Cloth Simulation Filter ground classification
- [SciPy](https://scipy.org/) — KD-Tree search, least-squares fitting, interpolation
- [NumPy](https://numpy.org/)
- [rasterio](https://rasterio.readthedocs.io/) — DEM (GeoTIFF) sampling
- [GeoPandas](https://geopandas.org/) — road-edge vector data handling

All dependencies are containerized — you do not need to install them manually if using Docker.

## Getting Started (Docker)

A `docker-compose.yaml` is provided for a reproducible environment.

```bash
# Build and start the container
docker compose up -d --build

# Attach a shell inside the container
docker compose exec 3d-geo-aligner bash
```

The repository is mounted at `/app` inside the container.

## Usage

### 1. Registration

Register a source LAS file against a target (adjacent, overlapping) LAS file:

```bash
python scripts/registration.py data/scaniverse_tondabayashi_01.las \
                                data/scaniverse_tondabayashi_02.las \
                                --out_dir demo_01_02
```

Results (registered LAS, transformation matrix, overlay images, and an execution log) are written to `results/demo_01_02/`.

Key optional parameters (see `python scripts/registration.py --help` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--voxel_size` | `0.005` | Downsampling voxel size (m) |
| `--pixel_size` | `0.045` | 2D projection resolution (m/pixel) |
| `--overlap_threshold` | `1.0` | XY distance threshold for overlap extraction (m) |
| `--csf_cloth_res` / `--csf_rigidness` / `--csf_class_threshold` | `1.0` / `3` / `0.5` | CSF ground-filtering parameters |
| `--slice_h_min` / `--slice_h_max` / `--slice_step` / `--slice_thickness` | `0.7` / `2.5` / `0.2` / `0.2` | Height range searched for the best wall slice used in matching (m) |
| `--orb_nfeatures` / `--ransac_iter` / `--ransac_threshold` / `--lowe_ratio` | `5000` / `10000` / `3.0` / `0.75` | ORB + RANSAC matching parameters |
| `--icp_threshold_factor` / `--icp_max_iter` | `0.75` / `2000` | 2D ICP refinement parameters |
| `--no_log` | off | Suppress overlay images / log output |

To reconstruct the full demo area, chain the registration pairwise (e.g. register `01→02`, then feed the resulting aligned cloud in as the new source for `02→03` registration, and so on along the T-shaped route).

### 2. Georeferencing

Once point clouds are registered/integrated into a single local coordinate frame, assign absolute coordinates using open reference data (road-edge vectors and a DEM):

```bash
python scripts/georeference.py <integrated_point_cloud.las> \
                                --road_edge <road_edge.shp> \
                                --dem <dem.tif> \
                                --out_dir demo_georef
```

> Road-edge and DEM reference data can be obtained free of charge from your national mapping agency (in Japan: GSI's Fundamental Geospatial Data). Convert road-edge data to Shapefile and DEM data to GeoTIFF before use, as described in the paper. Run `python scripts/georeference.py --help` for the exact set of available options.

## Method Summary

**Registration**
1. Extract mutually overlapping regions between source and target clouds via 2D (XY) nearest-neighbor search.
2. Classify ground / non-ground points using CSF.
3. **Horizontal:** slice non-ground points at increasing relative heights, orthographically project to a binary image, match via ORB + RANSAC, then refine with a Z-flattened 2D ICP.
4. **Vertical:** fit a first-order tilt plane (`Δz ≈ a·x + b·y + c`) between horizontally-aligned source ground points and target ground points via least squares, then apply the correction to the full point cloud.

**Georeferencing**
1. Classify ground / non-ground points using CSF.
2. **Horizontal:** project a non-ground wall slice to a binary image, generate a distance-transform similarity score image from road-edge vector data, and search for the best-matching translation/rotation via normalized cross-correlation template matching.
3. **Vertical:** compute the elevation offset between ground points and a DEM on a coarse grid, gap-fill and smooth it, then apply the correction via bilinear interpolation — preserving fine-scale terrain detail while fitting the global elevation trend to the DEM.

## Citation

If you use this pipeline in your research, please cite:

```
Ueda, R., Yoshida, D., 2026. A Pipeline for Low-Cost Wide-Area 3D Mapping Using
LiDAR-Equipped Mobile Devices and Open Data. ISPRS.
```

## Acknowledgements

We thank Tondabayashi City, Osaka Prefecture, Japan, for providing access to experimental sites. This research was supported by the Japan Science and Technology Agency (JST), Co-Creation Field Formation Support Program (grant no. JPMJPF2115).

## License

Add your chosen license here (e.g. MIT, Apache-2.0) before publishing.