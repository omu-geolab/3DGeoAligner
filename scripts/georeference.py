"""
Automatic georeferencing of LAS point clouds
"""

import argparse
import numpy as np
import laspy as lp
from pyproj import CRS, Transformer
import geopandas as gpd
import cv2
import open3d as o3d
import CSF
from shapely.geometry import box, LineString, MultiLineString, Polygon, MultiPolygon
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import os
import datetime
import rasterio
from scipy.interpolate import NearestNDInterpolator


class AutoGeoreferencer:
    """Executes automatic georeferencing of LAS point clouds."""
    def __init__(self, pixel_size):
        self.pixel_size  = pixel_size
        self.grid_min_x  = 0
        self.grid_max_y  = 0
        self.img_width   = 0
        self.img_height  = 0

    def load_las(self, path):
        print(f":: Loading LAS file: {path}")
        if not os.path.exists(path):
            print(f"!! Error: File not found -> {path}")
            return None, None
        try:
            las = lp.read(path)
        except Exception as e:
            print(f"!! Error reading LAS: {e}")
            return None, None

        points = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd, las

    def convert_crs(self, las_data, pcd, src_crs, tgt_crs, target_scale=0.001):
        print(f":: Converting LAS coordinates from {src_crs.name} to {tgt_crs.name}")
        try:
            crs_transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        except Exception as e:
            print(f"!! Error creating CRS transformer: {e}")
            return None, None

        new_x, new_y = crs_transformer.transform(las_data.x, las_data.y)
        original_z   = las_data.z.copy()

        las_data.header.offsets = np.array([np.min(new_x), np.min(new_y), np.min(original_z)])
        las_data.header.scales  = np.array([target_scale, target_scale, target_scale])
        las_data.x = new_x
        las_data.y = new_y
        las_data.z = original_z

        pcd.points = o3d.utility.Vector3dVector(np.vstack((new_x, new_y, original_z)).T)
        return las_data, pcd

    def setup_grid(self, las_data, buffer):
        las_x_min, las_x_max = np.min(las_data.x), np.max(las_data.x)
        las_y_min, las_y_max = np.min(las_data.y), np.max(las_data.y)

        self.grid_min_x = las_x_min - buffer
        grid_max_x      = las_x_max + buffer
        grid_min_y      = las_y_min - buffer
        self.grid_max_y = las_y_max + buffer

        self.img_width  = int(np.ceil((grid_max_x - self.grid_min_x) / self.pixel_size))
        self.img_height = int(np.ceil((self.grid_max_y - grid_min_y)  / self.pixel_size))

        print(f"   -> [Grid] {self.img_width} x {self.img_height} px  ({self.pixel_size} m/px)")
        return box(self.grid_min_x, grid_min_y, grid_max_x, self.grid_max_y)

    def world_to_pixel(self, x, y):
        px = (x - self.grid_min_x) / self.pixel_size
        py = (self.grid_max_y - y) / self.pixel_size
        return px.astype(np.int32), py.astype(np.int32)

    def rasterize_shp(self, gdf, grid_bbox, line_thickness):
        print(":: Processing Shapefile features")
        try:
            candidate_indices = list(gdf.sindex.intersection(grid_bbox.bounds))
            clipped_gdf = gdf.iloc[candidate_indices].clip(grid_bbox)
        except Exception as e:
            print(f"!! Error clipping Shapefile: {e}")
            return None

        if len(clipped_gdf) == 0:
            print("!! ERROR: No road features found in the LAS area")
            return None

        shp_img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        for geom in clipped_gdf.geometry:
            line_geometries = []
            if isinstance(geom, LineString):
                line_geometries.append(geom)
            elif isinstance(geom, MultiLineString):
                line_geometries.extend(geom.geoms)
            elif isinstance(geom, Polygon):
                line_geometries.append(geom.boundary)
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    line_geometries.append(poly.boundary)

            for line_geom in line_geometries:
                if line_geom.is_empty:
                    continue
                xs, ys = line_geom.xy
                px, py = self.world_to_pixel(np.array(xs), np.array(ys))
                if np.any((px >= 0) & (px < self.img_width) & (py >= 0) & (py < self.img_height)):
                    cv2.polylines(shp_img, [np.column_stack([px, py])], isClosed=False, color=255, thickness=line_thickness)

        return shp_img

    def rasterize_las_slice(self, pcd, cloth_resolution, rigidness, class_threshold, h_min, h_max, dilate_iter, close_kernel):
        print(":: Filtering ground (CSF) and slicing walls")
        csf = CSF.CSF()
        csf.params.cloth_resolution = cloth_resolution
        csf.params.rigidness = rigidness
        csf.params.class_threshold  = class_threshold
        csf.setPointCloud(np.asarray(pcd.points))
        ground_idx     = CSF.VecInt()
        nonground_idx  = CSF.VecInt()
        csf.do_filtering(ground_idx, nonground_idx)

        ground    = pcd.select_by_index(list(ground_idx))
        nonground = pcd.select_by_index(list(nonground_idx))

        ground_pts    = np.asarray(ground.points)
        nonground_pts = np.asarray(nonground.points)

        if len(ground_pts) == 0:
            return None

        tree = cKDTree(ground_pts[:, :2])
        _, idx = tree.query(nonground_pts[:, :2], k=1)
        ground_z = ground_pts[idx, 2]
        rel_z = nonground_pts[:, 2] - ground_z
        
        mask = (rel_z >= h_min) & (rel_z < h_max)
        slice_pts = nonground_pts[mask]

        if len(slice_pts) == 0:
            return None

        las_img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        px, py  = self.world_to_pixel(slice_pts[:, 0], slice_pts[:, 1])
        valid   = (px >= 0) & (px < self.img_width) & (py >= 0) & (py < self.img_height)
        las_img[py[valid], px[valid]] = 255

        las_img = cv2.dilate(las_img, np.ones((3, 3), np.uint8), iterations=dilate_iter)
        las_img = cv2.morphologyEx(las_img, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
        return las_img

    def estimate_transform(self, las_img, shp_img, alpha, search_range_m, angle_min, angle_max, angle_step, min_score):
        print(":: Computing alignment")
        las_img_blur   = cv2.GaussianBlur(las_img, (5, 5), 0)
        dist_img       = 1.0 / (1.0 + cv2.distanceTransform(255 - shp_img, cv2.DIST_L2, 5) * alpha)
        shp_score_img  = cv2.normalize(dist_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        pad_px          = int(np.ceil(search_range_m / self.pixel_size))
        shp_img_padded  = cv2.copyMakeBorder(shp_score_img, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=0)

        best_score = -1
        best_M = None
        h, w   = las_img.shape
        center = (w // 2, h // 2)

        for angle in np.arange(angle_min, angle_max + angle_step * 0.5, angle_step):
            rot_matrix    = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated       = cv2.warpAffine(las_img_blur, rot_matrix, (w, h))
            match_result  = cv2.matchTemplate(shp_img_padded, rotated, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match_result)

            if max_val > best_score:
                best_score     = max_val
                dx = max_loc[0] - pad_px
                dy = max_loc[1] - pad_px
                candidate_matrix = rot_matrix.copy()
                candidate_matrix[0, 2] += dx
                candidate_matrix[1, 2] += dy
                best_M = candidate_matrix

        if best_M is None or best_score < min_score:
            print(f"!! Alignment score too low or failed (Score: {best_score:.4f})")
            return None

        print(f"   -> Best Match Score: {best_score:.4f}")
        return best_M

    def apply_xy_transform(self, las_data, transform_matrix):
        print(":: Applying affine transformation (XY) to LAS coordinates")
        px    = (las_data.x - self.grid_min_x) / self.pixel_size
        py    = (self.grid_max_y - las_data.y)  / self.pixel_size
        ones  = np.ones(len(px))
        transformed_pixels = transform_matrix @ np.vstack((px, py, ones))
        las_data.x = transformed_pixels[0] * self.pixel_size + self.grid_min_x
        las_data.y = self.grid_max_y - transformed_pixels[1] * self.pixel_size
        return las_data

    def align_z_with_dem(self, las_data, dem_path, current_crs, smooth_kernel_size, grid_res, max_ground_samples, dem_valid_min=-1000, dem_valid_max=4000):
        print(f":: Starting smooth Z-alignment using DEM: {dem_path}")

        if not os.path.exists(dem_path):
            print("!! Error: DEM file not found")
            return None, None, None

        las_x_min, las_x_max = np.min(las_data.x), np.max(las_data.x)
        las_y_min, las_y_max = np.min(las_data.y), np.max(las_data.y)

        print("   -> Detecting ground points for difference calculation")
        all_pts = np.vstack((las_data.x, las_data.y, las_data.z)).T
        if len(all_pts) > max_ground_samples:
            all_pts = all_pts[np.random.choice(len(all_pts), max_ground_samples, replace=False)]

        csf = CSF.CSF()
        csf.params.cloth_resolution = 1.0
        csf.params.class_threshold  = 0.5
        csf.setPointCloud(np.asarray(all_pts))
        ground_idx    = CSF.VecInt()
        nonground_idx = CSF.VecInt()
        csf.do_filtering(ground_idx, nonground_idx)

        if len(ground_idx) == 0:
            print("!! Warning: No ground points found. Skipping Z-alignment")
            return None, None, None

        ground_pts = all_pts[list(ground_idx)]

        print("   -> Sampling DEM heights")
        try:
            with rasterio.open(dem_path) as dem_src:
                if current_crs != dem_src.crs:
                    crs_transformer = Transformer.from_crs(current_crs, dem_src.crs, always_xy=True)
                    ground_x, ground_y = crs_transformer.transform(ground_pts[:, 0], ground_pts[:, 1])
                else:
                    ground_x, ground_y = ground_pts[:, 0], ground_pts[:, 1]
                dem_z = np.array([v[0] for v in dem_src.sample(zip(ground_x, ground_y))])
        except Exception as e:
            print(f"!! Error reading DEM: {e}")
            return None, None, None

        valid_dem  = (dem_z > dem_valid_min) & (dem_z < dem_valid_max)
        ground_pts = ground_pts[valid_dem]
        dem_z      = dem_z[valid_dem]

        if len(ground_pts) == 0:
            print("!! Warning: No ground points found. Skipping Z-alignment")
            return None, None, None

        z_diffs = dem_z - ground_pts[:, 2]

        grid_w = int(np.ceil((las_x_max - las_x_min) / grid_res)) + 1
        grid_h = int(np.ceil((las_y_max - las_y_min) / grid_res)) + 1
        print(f"   -> Creating adjustment grid ({grid_w}x{grid_h}, {grid_res} m/cell)")

        grid_sum   = np.zeros((grid_h, grid_w), dtype=np.float32)
        grid_count = np.zeros((grid_h, grid_w), dtype=np.float32)

        idx_x     = ((ground_pts[:, 0] - las_x_min) / grid_res).astype(np.int32)
        idx_y     = ((las_y_max - ground_pts[:, 1]) / grid_res).astype(np.int32)
        valid_idx = (idx_x >= 0) & (idx_x < grid_w) & (idx_y >= 0) & (idx_y < grid_h)

        np.add.at(grid_sum,   (idx_y[valid_idx], idx_x[valid_idx]), z_diffs[valid_idx])
        np.add.at(grid_count, (idx_y[valid_idx], idx_x[valid_idx]), 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            z_adjustment_grid = grid_sum / grid_count

        y_valid, x_valid = np.where(grid_count > 0)
        if len(y_valid) > 0:
            nn_interp = NearestNDInterpolator(
                np.column_stack((x_valid, y_valid)), z_adjustment_grid[y_valid, x_valid])
            Y, X = np.mgrid[0:grid_h, 0:grid_w]
            z_adjustment_grid = nn_interp((X, Y))
        else:
            z_adjustment_grid = np.zeros_like(z_adjustment_grid)

        k_size = smooth_kernel_size | 1
        print(f"   -> Smoothing adjustment grid (kernel: {k_size}x{k_size})")
        smoothed_z_adjustment = cv2.GaussianBlur(z_adjustment_grid, (k_size, k_size), 0)

        ground_pts_before = ground_pts.copy()
        print("   -> Applying smooth Z adjustment to all points")
        grid_interp = RegularGridInterpolator((np.arange(grid_h), np.arange(grid_w)), smoothed_z_adjustment, bounds_error=False, fill_value=None)
        all_x_idx = (las_data.x - las_x_min) / grid_res
        all_y_idx = (las_y_max - las_data.y)  / grid_res
        las_data.z += grid_interp(np.column_stack((all_y_idx, all_x_idx)))
        print("   -> Z-alignment complete")

        ground_pts_after = ground_pts.copy()
        ground_pts_after[:, 2] += grid_interp(np.column_stack(((las_y_max - ground_pts[:, 1]) / grid_res, (ground_pts[:, 0] - las_x_min) / grid_res)))
        return ground_pts_before, ground_pts_after, dem_z

    def save_las(self, las_data, output_path, tgt_crs, target_scale=0.001):
        print(f":: Saving LAS file -> {output_path}")
        header = lp.LasHeader(point_format=las_data.header.point_format,version=las_data.header.version)
        header.scales  = np.array([target_scale, target_scale, target_scale])
        header.offsets = np.array([np.min(las_data.x), np.min(las_data.y), np.min(las_data.z)])
        try:
            header.add_crs(tgt_crs)
        except Exception:
            pass

        new_las = lp.LasData(header)
        new_las.x = las_data.x
        new_las.y = las_data.y
        new_las.z = las_data.z

        for dim in las_data.point_format.dimension_names:
            if dim not in ('X', 'Y', 'Z'):
                try:
                    new_las[dim] = las_data[dim]
                except Exception:
                    pass

        new_las.write(output_path)
        print(f"   -> Saved: {output_path}")

        
class GeoreferencingExporter:
    """Handles overlay image and log output for a georeferencing run."""
    def __init__(self, log_dir, base_filename, pixel_size):
        self.log_dir    = log_dir
        self.base_filename = base_filename
        self.pixel_size    = pixel_size

    def save_overlay(self, las_img, shp_img, suffix, transform_matrix=np.eye(2,3)):
        h, w = shp_img.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 2] = shp_img
        overlay[:, :, 1] = cv2.warpAffine(las_img, transform_matrix, (w, h))
        path = os.path.join(self.log_dir, f"{self.base_filename}_{suffix}_overlay.png")
        cv2.imwrite(path, overlay)
        print(f"   -> Saved: {path}")
    
    def save_dem_ground_overlay(self, ground_pts, dem_z, suffix):
        if ground_pts is None or dem_z is None or len(ground_pts) == 0:
            print(f"!! Skipping DEM comparison image (no data): {suffix}")
            return

        all_x   = ground_pts[:, 0]
        las_z   = ground_pts[:, 2]

        x_min, x_max = all_x.min(),  all_x.max()
        z_min = min(las_z.min(), dem_z.min()) - 1.0
        z_max = max(las_z.max(), dem_z.max()) + 1.0

        scale = 1.0 / self.pixel_size
        w = max(int(np.ceil((x_max - x_min) * scale)) + 1, 1)
        h = max(int(np.ceil((z_max - z_min) * scale)) + 1, 1)

        img = np.zeros((h, w, 3), dtype=np.uint8)

        def plot_pts(xs, zs, channel):
            x_px = ((xs - x_min) * scale).astype(np.int32)
            z_px = ((zs - z_min) * scale).astype(np.int32)
            valid = (x_px >= 0) & (x_px < w) & (z_px >= 0) & (z_px < h)
            img[h - z_px[valid] - 1, x_px[valid], channel] = 255

        plot_pts(all_x, dem_z, 2)

        plot_pts(all_x, las_z, 1)

        path = os.path.join(self.log_dir, f"{self.base_filename}_{suffix}.png")
        cv2.imwrite(path, img)
        print(f"   -> Saved: {path}")

    def write_execution_log(self, args, timestamp, process_time, transform_matrix, tgt_epsg):
        log_path    = os.path.join(self.log_dir, f"{self.base_filename}_log.txt")
        matrix_path = os.path.join(self.log_dir, f"{self.base_filename}_matrix.txt")
        print(f":: Writing execution log: {log_path}")

        dx_m    = transform_matrix[0, 2] * args.pixel_size
        dy_m    = transform_matrix[1, 2] * args.pixel_size
        rot_deg = np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]))

        with open(log_path, "w") as f:
            f.write("=== Georeference Log ===\n")
            f.write(f"Timestamp          : {timestamp}\n")
            f.write(f"Process Time  : {process_time}\n")
            f.write(f"LAS                : {args.las_path}\n")
            f.write(f"SHP                : {args.shp_path}\n")
            f.write(f"DEM                : {args.dem_path}\n")
            f.write(f"Target CRS (EPSG)  : {tgt_epsg}\n\n")
            f.write("--- Parameters ---\n")
            for key, val in vars(args).items():
                f.write(f"{key:<25}: {val}\n")
            f.write("\n--- XY Correction Results ---\n")
            f.write(f"XY Correction X    : {dx_m:.4f} m\n")
            f.write(f"XY Correction Y    : {dy_m:.4f} m\n")
            f.write(f"Rotation           : {rot_deg:.4f} degrees\n")
            f.write(f"Matrix file        : {os.path.basename(matrix_path)}\n")
        print(f"   -> Saved: {log_path}")

        np.savetxt(matrix_path, transform_matrix, fmt="%.10f", delimiter=",")
        print(f"   -> Saved: {matrix_path}")


def georeference_main():
    parser = argparse.ArgumentParser(description="Auto-georeference a LAS file to a road-edge SHP and DEM")
 
    # --- Input / Output ---
    parser.add_argument("las_path", help="Input LAS file path")
    parser.add_argument("shp_path", help="Reference road-edge SHP file path")
    parser.add_argument("dem_path", help="Reference DEM GeoTIFF path")
    parser.add_argument("--out_dir", type=str, default=None, help="Output subdirectory inside results/")
 
    # --- CRS ---
    parser.add_argument("--src_epsg", type=int,   default=32653, help="EPSG code of input LAS (default: 32653)")
    parser.add_argument("--tgt_epsg", type=int,   default=None,  help="Force target EPSG (default: auto from SHP)")
 
    # --- Preprocessing ---
    parser.add_argument("--pixel_size",     type=float, default=0.1,  help="Raster projection resolution in meters/pixel (default: 0.1)")
    parser.add_argument("--buffer",         type=float, default=20.0, help="Grid buffer in meters (default: 20.0)")
    parser.add_argument("--line_thickness", type=int,   default=3,    help="SHP line thickness in pixels (default: 3)")
 
    # --- CSF-based ground filtering ---
    parser.add_argument("--cloth_resolution", type=float, default=1.0, help="CSF cloth resolution (default: 1.0)")
    parser.add_argument("--rigidness",        type=int,   default=3,   help="CSF rigidness parameter (default: 3)")
    parser.add_argument("--class_threshold",  type=float, default=0.5, help="CSF classification threshold (default: 0.5)")
    parser.add_argument("--h_min",            type=float, default=0.5, help="Minimum relative height of wall slice in meters (default: 0.5)")
    parser.add_argument("--h_max",            type=float, default=2.5, help="Maximum relative height of wall slice in meters (default: 2.5)")
    parser.add_argument("--dilate_iter",      type=int,   default=1,   help="Number of dilation iterations for wall slice image (default: 1)")
    parser.add_argument("--close_kernel",     type=int,   default=7,   help="Kernel size for morphological closing of wall slice image (default: 7)")
 
    # --- Template matching-based XY alignment ---
    parser.add_argument("--decay_coefficient", type=float, default=0.10, help="Distance decay coefficient for SHP score image (default: 0.10)")
    parser.add_argument("--search_range",      type=float, default=5.0,  help="XY search range in meters (default: 5.0)")
    parser.add_argument("--angle_min",         type=float, default=-5.0, help="Rotation search min in degrees (default: -5.0)")
    parser.add_argument("--angle_max",         type=float, default=5.0,  help="Rotation search max in degrees (default: 5.0)")
    parser.add_argument("--angle_step",        type=float, default=0.25, help="Rotation search step in degrees (default: 0.25)")
    parser.add_argument("--min_score",         type=float, default=0.15, help="Minimum alignment score threshold (default: 0.15)")
 
    # --- DEM-based Z alignment ---
    parser.add_argument("--smooth_kernel",      type=int,   default=51,     help="Gaussian kernel size for Z smoothing (default: 51)")
    parser.add_argument("--grid_res",           type=float, default=2.0,    help="Adjustment grid resolution in meters (default: 2.0)")
    parser.add_argument("--max_ground_samples", type=int,   default=200000, help="Max ground points sampled for Z alignment (default: 200000)")
 
    # --- Log ---
    parser.add_argument("--no_log", action="store_true", help="Suppress all intermediate and final image/log outputs (default: outputs enabled)")
    args = parser.parse_args()

    timestamp  = datetime.datetime.now().strftime("%Y%m%d")
    start_time = datetime.datetime.now()

    output_dir = os.path.join(f"results", args.out_dir) if args.out_dir else f"results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    max_index = 0
    for entry in os.listdir(output_dir):
        if entry.startswith("g") and os.path.isdir(os.path.join(output_dir, entry)):
            try:
                idx = int(entry.replace("g", ""))
                if idx > max_index:
                    max_index = idx
            except ValueError:
                pass

    new_index = max_index + 1
    log_dir = os.path.join(output_dir, f"g{new_index}_log")
    os.makedirs(log_dir, exist_ok=True)
    output_las_path = os.path.join(output_dir, f"g{new_index}.las")
    base_filename = f"g{new_index}"
    print(f":: Output directory: {os.path.abspath(output_dir)}")

    georeferencer = AutoGeoreferencer(pixel_size=args.pixel_size)

    pcd, las = georeferencer.load_las(args.las_path)
    if las is None:
        return

    las_crs = las.header.parse_crs()
    tgt_crs = None
    src_crs = None
    if las_crs is not None:
        print(f"   -> Auto-detected CRS from LAS: {las_crs.name}")
        src_crs = las_crs
    else:
        print(f"   -> No CRS info in LAS header. Using fallback (--src_crs): EPSG:{args.src_epsg}")
        src_crs = CRS.from_user_input(args.src_epsg)
        
    print(f":: Reading SHP file: {args.shp_path}")
    try:
        gdf = gpd.read_file(args.shp_path)
    except Exception as e:
        print(f"!! Error reading SHP file: {e}")
        return

    tgt_crs = None
    if args.tgt_epsg is not None:
        tgt_crs = CRS.from_epsg(args.tgt_epsg)
        gdf.set_crs(tgt_crs, allow_override=True, inplace=True)
    elif gdf.crs is not None:
        tgt_crs = gdf.crs
        print(f"   -> Target CRS detected: {tgt_crs.name}")
    else:
        print("!! ERROR: No CRS detected. Use --tgt_epsg")
        return
    
    las, pcd = georeferencer.convert_crs(las, pcd, src_crs=src_crs, tgt_crs=tgt_crs)
    if las is None:
        return

    grid_bbox = georeferencer.setup_grid(las, buffer=args.buffer)
    shp_img   = georeferencer.rasterize_shp(gdf, grid_bbox, args.line_thickness)
    las_img   = georeferencer.rasterize_las_slice(
        pcd, 
        cloth_resolution=args.cloth_resolution, 
        rigidness=args.rigidness, 
        class_threshold=args.class_threshold, 
        h_min=args.h_min, h_max=args.h_max, 
        dilate_iter=args.dilate_iter, 
        close_kernel=args.close_kernel
        )

    if shp_img is None or las_img is None:
        print("!! Error generating process images")
        return

    transform_matrix = georeferencer.estimate_transform(
        las_img, shp_img, 
        alpha=args.decay_coefficient, 
        search_range_m=args.search_range, 
        angle_min=args.angle_min, 
        angle_max=args.angle_max, 
        angle_step=args.angle_step, 
        min_score=args.min_score
        )

    if transform_matrix is None:
        print("!! Alignment failed")
        return

    las = georeferencer.apply_xy_transform(las, transform_matrix)

    ground_pts_before, ground_pts_after, dem_z = georeferencer.align_z_with_dem(
        las, args.dem_path, 
        tgt_crs, smooth_kernel_size=args.smooth_kernel, 
        grid_res=args.grid_res, 
        max_ground_samples=args.max_ground_samples
        )

    georeferencer.save_las(las, output_las_path, tgt_crs)
    end_time = datetime.datetime.now()
    process_time =  end_time - start_time
    
    if not args.no_log:
        print(":: Generating result overlay images and logs")

        exporter = GeoreferencingExporter(log_dir, base_filename, args.pixel_size)

        exporter.save_overlay(las_img, shp_img, suffix="01_before")
        exporter.save_overlay(las_img, shp_img, suffix="02_after", transform_matrix=transform_matrix)
        exporter.save_dem_ground_overlay(ground_pts_before, dem_z, suffix="04_dem_before")
        exporter.save_dem_ground_overlay(ground_pts_after,  dem_z, suffix="05_dem_after")
        exporter.write_execution_log(args, timestamp, process_time, transform_matrix, args.tgt_epsg)

if __name__ == "__main__":
    georeference_main()