# 3D Geo Aligner

**A low-cost pipeline for wide-area 3D mapping using LiDAR-equipped mobile devices and open data.**

This repository provides the reference implementation of the method described in:

> Ueda, R., Yoshida, D. *A Pipeline for Low-Cost Wide-Area 3D Mapping Using LiDAR-Equipped Mobile Devices and Open Data.* (ISPRS, 2026)

The pipeline integrates multiple locally captured point clouds into a single, georeferenced 3D map. The process requires only mobile devices (e.g. iPhone Pro/iPad) and open geospatial data (Road Edge Data and 5m DEM).

The core idea is to **decompose 3D registration and georeferencing into independent horizontal and vertical steps**, which reduces errors and improve computational efficiency.

## Pipeline Overview

| Stage | Script | Description |
|---|---|---|
| **Registration** | `scripts/registration.py` | Aligns a source point cloud to a target (adjacent) point cloud. |
| **Georeferencing** | `scripts/georeference.py` | Assigns absolute coordinates to the registered/integrated point cloud. |


## Repository Structure

```
.
├── docker-compose.yaml
├── Dockerfile
├── results
├── scripts/
│   ├── registration.py       # Point cloud ↔ point cloud registration
│   └── georeference.py       # Point cloud ↔ open data georeferencing
└── data/
    ├── scaniverse_tondabayashi_01.las
    ├── scaniverse_tondabayashi_02.las
    ├── scaniverse_tondabayashi_03.las
    ├── scaniverse_tondabayashi_04.las
    ├── scaniverse_tondabayashi_05.las
    └── scaniverse_tondabayashi_01-05.las
```

## Demo Data

Five sample scans captured with an iPad Pro (LiDAR) using the **Scaniverse** app are provided in `data/` for trying out the pipeline end to end. The scans were collected along adjacent, overlapping routes in the Jinaimachi district, Tondabayashi City, Osaka, Japan, and connect in a **T-shape** layout:

```
               Scan 5
                 |
Scan 1 — Scan 2 — — Scan 3 — Scan 4
```

Adjacent, overlapping pairs suitable for registration are:

- `01` ↔ `02`
- `02` ↔ `03`
- `03` ↔ `04`
- `02` ↔ `05`
- `03` ↔ `05`

These pairs can be chained to reconstruct the full T-shaped area (e.g. register `01`→`02`, then use the result as the new target for `03`, and so on).

For the georeferencing demo, the following GSI (Geospatial Information Authority of Japan) Fundamental Geospatial Data products were obtained and converted before use:

- `FG-GML-5135-54-98-DEM5A-20250711.tif.aux.xml` (5m DEM) → converted to a GeoTIFF (`.tif`) and used as the `dem_path` reference input.
- `FG-GML-513554-RdEdg-20260401-0001.xml` (road edge data) → converted to a Shapefile (`.shp`) and used as the `shp_path` reference input.

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
docker-compose up -d --build

# Attach a shell inside the container
docker-compose exec 3d-geo-aligner bash
```

The repository is mounted at `/app` inside the container.

## Output

Every run writes its results under `results/`.

- If `--out_dir <name>` is given, output goes to `results/<name>/`.
- Otherwise, output goes to `results/<YYYYMMDD>/` (today's date), so multiple runs on the same day share a dated folder.

Within that folder, each run is numbered automatically by scanning existing output for the highest existing index and incrementing it (so re-running against the same `--out_dir`/date accumulates numbered attempts rather than overwriting):

- **Registration** writes the aligned point cloud as `r[n].las`, where `[n]` is the attempt number (e.g. `r1.las`, `r2.las`, ...).
- **Georeferencing** writes the aligned point cloud as `g[n].las` (e.g. `g1.las`, `g2.las`, ...).

Unless `--no_log` is passed, each run also writes a companion log folder alongside the `.las` file — `r[n]_log/` for registration or `g[n]_log/` for georeferencing — containing the transformation matrix, overlay images, and an execution log (parameters used, timing, and success/failure details). Passing `--no_log` suppresses this folder and its contents entirely.

## Usage

### 1. Registration

Register a source LAS file against a target (adjacent, overlapping) LAS file:

```bash
python scripts/registration.py data/scaniverse_tondabayashi_01.las \
                                data/scaniverse_tondabayashi_02.las \
                                --out_dir demo_01_02
```

Results (registered LAS as `r[n].las`, transformation matrix, overlay images, and an execution log) are written to `results/demo_01_02/` (see [Output](#output) for the full naming convention).

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
| `--no_log` | off | Suppress the `r[n]_log/` overlay images / execution log output |

To reconstruct the full demo area, chain the registration pairwise (e.g. register `01→02`, then feed the resulting aligned cloud in as the new source for `02→03` registration, and so on along the T-shaped route).

### 2. Georeferencing

Once point clouds are registered/integrated into a single local coordinate frame, assign absolute coordinates using open reference data (road-edge vectors and a DEM). `las_path`, `road_edge.shp`, and `dem.tif` are positional arguments, in that order:

```bash
python scripts/georeference.py data/scaniverse_tondabayashi_01-05.las \
                                <road_edge.shp> \
                                <dem.tif> \
                                --out_dir demo_georef
```

Road-edge and DEM reference data can be obtained free of charge from your national mapping agency (in Japan: GSI's Fundamental Geospatial Data). Convert road-edge data to Shapefile and DEM data to GeoTIFF before use (see [Demo Data](#demo-data) for the exact source files used in this demo). Run `python scripts/georeference.py --help` for the exact set of available options.

Results (georeferenced LAS as `g[n].las`, transformation matrix, overlay images, and an execution log) are written to `results/demo_georef/` (see [Output](#output) for the full naming convention).

Key optional parameters (see `python scripts/georeference.py --help` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--epsg` | `32653` | EPSG code to reproject and process everything in |
| `--pixel_size` / `--buffer` / `--line_thickness` | `0.1` / `20.0` / `3` | Rasterization settings for the road-edge / wall-slice images |
| `--cloth_resolution` / `--rigidness` / `--class_threshold` | `1.0` / `3` / `0.5` | CSF ground-filtering parameters |
| `--h_min` / `--h_max` | `0.5` / `2.5` | Relative height range of the wall slice used for horizontal alignment (m) |
| `--dilate_iter` / `--close_kernel` | `1` / `7` | Morphological smoothing of the wall-slice image |
| `--decay_coefficient` / `--search_range` / `--angle_min` / `--angle_max` / `--angle_step` / `--min_score` | `0.10` / `5.0` / `-5.0` / `5.0` / `0.25` / `0.15` | Template-matching search parameters for horizontal alignment |
| `--smooth_kernel` / `--grid_res` / `--max_ground_samples` | `51` / `2.0` / `200000` | DEM-based vertical (Z) alignment parameters |
| `--no_log` | off | Suppress the `g[n]_log/` overlay images / execution log output |

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