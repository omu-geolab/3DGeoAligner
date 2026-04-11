import argparse
import numpy as np
import laspy
from pyproj import CRS, Transformer
import geopandas as gpd
import cv2
import open3d as o3d
import CSF
from shapely.geometry import box, LineString, MultiLineString, Polygon, MultiPolygon
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import os
import sys
import datetime
import rasterio
from scipy.interpolate import NearestNDInterpolator

class AutoGeoreferencer:
    def __init__(self, pixel_size=0.1):
        self.pixel_size = pixel_size
        self.min_x = 0
        self.max_y = 0
        self.img_width = 0
        self.img_height = 0

    # -------------------------------------------------------------------------
    # 1. LASデータの読み込み
    # -------------------------------------------------------------------------
    def load_las(self, path):
        print(f":: Loading LAS file: {path}")
        if not os.path.exists(path):
            print(f"!! Error: File not found -> {path}")
            return None, None

        try:
            las = laspy.read(path)
        except Exception as e:
            print(f"!! Error reading LAS: {e}")
            return None, None

        points = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return las, pcd

    # -------------------------------------------------------------------------
    # LASデータの座標系変換
    # -------------------------------------------------------------------------
    def convert_las_crs(self, las_data, pcd, src_epsg, target_crs):
        print(f":: Converting LAS coordinates from EPSG:{src_epsg} to Target CRS ({target_crs.name})...")
        
        try:
            transformer = Transformer.from_crs(CRS.from_epsg(src_epsg), target_crs, always_xy=True)
        except Exception as e:
            print(f"!! Error creating CRS transformer: {e}")
            return None, None
        
        new_x, new_y = transformer.transform(las_data.x, las_data.y)
        original_z = las_data.z.copy()

        las_data.header.offsets = np.array([np.min(new_x), np.min(new_y), np.min(original_z)])
        las_data.header.scales = np.array([0.001, 0.001, 0.001])
        
        las_data.x = new_x
        las_data.y = new_y
        las_data.z = original_z
        
        points = np.vstack((new_x, new_y, original_z)).T
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print("   -> Conversion complete. LAS data is now in Target CRS.")
        return las_data, pcd

    # -------------------------------------------------------------------------
    # 2. グリッド設定
    # -------------------------------------------------------------------------
    def setup_grid(self, las_data, buffer=20.0):
        x_min = np.min(las_data.x)
        x_max = np.max(las_data.x)
        y_min = np.min(las_data.y)
        y_max = np.max(las_data.y)

        self.min_x = x_min - buffer
        max_x = x_max + buffer
        min_y = y_min - buffer
        self.max_y = y_max + buffer 

        width_m = max_x - self.min_x
        height_m = self.max_y - min_y

        self.img_width = int(np.ceil(width_m / self.pixel_size))
        self.img_height = int(np.ceil(height_m / self.pixel_size))
        
        print(f"   [Grid Info] Size: {self.img_width} x {self.img_height} px (Resolution: {self.pixel_size}m/px)")
        
        return box(self.min_x, min_y, max_x, self.max_y)

    def world_to_pixel(self, x, y):
        px = (x - self.min_x) / self.pixel_size
        py = (self.max_y - y) / self.pixel_size
        return px.astype(np.int32), py.astype(np.int32)

    # -------------------------------------------------------------------------
    # 3. SHPファイルの処理
    # -------------------------------------------------------------------------
    def process_shp(self, gdf, bbox_poly):
        print(f":: Processing Shapefile features...")
        try:
            sindex = gdf.sindex
            possible_matches_index = list(sindex.intersection(bbox_poly.bounds))
            gdf_sub = gdf.iloc[possible_matches_index]
            clipped_gdf = gdf_sub.clip(bbox_poly)
        except Exception as e:
            print(f"!! Error clipping Shapefile: {e}")
            return None

        if len(clipped_gdf) == 0:
            print("!! ERROR: No road features found in the specified LAS area.")
            return None

        img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        
        for geom in clipped_gdf.geometry:
            draw_lines = []
            if isinstance(geom, (LineString, MultiLineString)):
                if isinstance(geom, LineString): draw_lines.append(geom)
                else: draw_lines.extend(geom.geoms)
            elif isinstance(geom, (Polygon, MultiPolygon)):
                if isinstance(geom, Polygon): draw_lines.append(geom.boundary)
                else: 
                    for poly in geom.geoms: draw_lines.append(poly.boundary)

            for line in draw_lines:
                if line.is_empty: continue
                xs, ys = line.xy
                px, py = self.world_to_pixel(np.array(xs), np.array(ys))
                
                if np.any((px >= 0) & (px < self.img_width) & (py >= 0) & (py < self.img_height)):
                    pts = np.column_stack([px, py])
                    cv2.polylines(img, [pts], isClosed=False, color=255, thickness=3)
            
        return img

    # -------------------------------------------------------------------------
    # 4. 点群スライス画像化
    # -------------------------------------------------------------------------
    def process_las_slice(self, pcd, h_min=1.0, h_max=1.5):
        print(":: Filtering ground (CSF) and slicing walls...")
        csf = CSF.CSF()
        csf.params.cloth_resolution = 1.0
        csf.params.class_threshold = 0.5
        csf.setPointCloud(np.asarray(pcd.points))
        ground_idx = CSF.VecInt()
        non_ground_idx = CSF.VecInt()
        csf.do_filtering(ground_idx, non_ground_idx)
        
        ground = pcd.select_by_index(list(ground_idx))
        non_ground = pcd.select_by_index(list(non_ground_idx))

        g_pts = np.asarray(ground.points)
        ng_pts = np.asarray(non_ground.points)
        
        if len(g_pts) == 0:
            return None

        tree = cKDTree(g_pts[:, :2])
        _, idx = tree.query(ng_pts[:, :2], k=1)
        ground_z = g_pts[idx, 2]
        rel_z = ng_pts[:, 2] - ground_z
        
        mask = (rel_z >= h_min) & (rel_z < h_max)
        sliced_points = ng_pts[mask]
        
        if len(sliced_points) == 0:
            return None

        img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        px, py = self.world_to_pixel(sliced_points[:, 0], sliced_points[:, 1])
        
        valid = (px >= 0) & (px < self.img_width) & (py >= 0) & (py < self.img_height)
        img[py[valid], px[valid]] = 255
        
        img = cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)
        kernel_close = np.ones((7, 7), np.uint8) 
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
        
        return img

    # -------------------------------------------------------------------------
    # 5. 位置合わせ計算
    # -------------------------------------------------------------------------
    def calculate_transform(self, img_las, img_shp):
        print(":: Computing alignment...")
        img_las_blur = cv2.GaussianBlur(img_las, (5, 5), 0)
        dist_transform = cv2.distanceTransform(255 - img_shp, cv2.DIST_L2, 5)
        dist_img = 1.0 / (1.0 + dist_transform * 0.1) 
        img_shp_score = cv2.normalize(dist_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        search_range_m = 5.0 
        pad_px = int(np.ceil(search_range_m / self.pixel_size))
        
        img_shp_padded = cv2.copyMakeBorder(
            img_shp_score, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=0
        )

        best_score = -1
        best_M = None
        h, w = img_las.shape
        center = (w // 2, h // 2)

        for angle in np.arange(-5.0, 5.25, 0.25):
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_las = cv2.warpAffine(img_las_blur, M_rot, (w, h))
            res = cv2.matchTemplate(img_shp_padded, rotated_las, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                dx = max_loc[0] - pad_px
                dy = max_loc[1] - pad_px
                M_best_candidate = M_rot.copy()
                M_best_candidate[0, 2] += dx
                M_best_candidate[1, 2] += dy
                best_M = M_best_candidate

        if best_M is None or best_score < 0.15:
            print(f"!! Alignment score too low or failed (Score: {best_score:.4f}).")
            return None

        print(f"   -> Best Match Score: {best_score:.4f}")
        return best_M

    def apply_xy_transform_memory(self, las_data, M_pixel):
        print(":: Applying affine transformation (XY) to LAS coordinates...")
        coords = np.vstack((las_data.x, las_data.y))
        px = (coords[0] - self.min_x) / self.pixel_size
        py = (self.max_y - coords[1]) / self.pixel_size
        ones = np.ones(coords.shape[1])
        p_coords = np.vstack((px, py, ones))
        trans_p_coords = M_pixel @ p_coords
        px_new = trans_p_coords[0]
        py_new = trans_p_coords[1]
        las_data.x = px_new * self.pixel_size + self.min_x
        las_data.y = self.max_y - py_new * self.pixel_size
        return las_data

    # -------------------------------------------------------------------------
    # 7. 【改】DEMによる滑らかなローカルZ座標補正
    # -------------------------------------------------------------------------
    def align_z_smoothly(self, las_data, dem_path, current_crs, smooth_kernel_size=51):
        """
        DEMとの差分グリッドを作成し、ガウスぼかしを適用して滑らかな補正曲面を作る。
        """
        print(f":: Starting Smooth Z-alignment using DEM: {dem_path}")
        
        if not os.path.exists(dem_path):
            print("!! Error: DEM file not found.")
            return

        # 1. LASの範囲確認
        x_min, x_max = np.min(las_data.x), np.max(las_data.x)
        y_min, y_max = np.min(las_data.y), np.max(las_data.y)

        # 2. 地面点を抽出 (CSF)
        print("   -> Detecting ground points for difference calculation...")
        # 処理速度のため、最大10万点程度にサンプリング
        points = np.vstack((las_data.x, las_data.y, las_data.z)).T
        if len(points) > 200000:
            indices = np.random.choice(len(points), 200000, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points

        csf = CSF.CSF()
        csf.params.cloth_resolution = 1.0
        csf.params.class_threshold = 0.5
        csf.setPointCloud(np.asarray(sample_points))
        ground_idx = CSF.VecInt()
        non_ground_idx = CSF.VecInt()
        csf.do_filtering(ground_idx, non_ground_idx)
        
        if len(ground_idx) == 0:
            print("!! Warning: No ground points found. Skipping Z-alignment.")
            return

        g_pts = sample_points[ground_idx] # Ground Points (x, y, z_las)

        # 3. DEMから標高取得
        print("   -> Sampling DEM heights...")
        dem_z_vals = []
        try:
            with rasterio.open(dem_path) as src:
                # 座標変換が必要な場合
                if current_crs != src.crs:
                    transformer = Transformer.from_crs(current_crs, src.crs, always_xy=True)
                    gx, gy = transformer.transform(g_pts[:, 0], g_pts[:, 1])
                    sample_coords = list(zip(gx, gy))
                else:
                    sample_coords = list(zip(g_pts[:, 0], g_pts[:, 1]))

                sampled = src.sample(sample_coords)
                for val in sampled:
                    dem_z_vals.append(val[0])
        except Exception as e:
            print(f"!! Error reading DEM: {e}")
            return

        dem_z_vals = np.array(dem_z_vals)
        
        # 無効値(-9999等)の除去
        valid_mask = (dem_z_vals > -1000) & (dem_z_vals < 4000)
        g_pts = g_pts[valid_mask]
        dem_z_vals = dem_z_vals[valid_mask]

        if len(g_pts) == 0:
            return

        # 差分（補正量） = DEM - LAS
        diffs = dem_z_vals - g_pts[:, 2]

        # 4. 補正量グリッドの作成 (2m解像度など少し粗くて良い)
        grid_res = 2.0  # 2m grid
        grid_w = int(np.ceil((x_max - x_min) / grid_res)) + 1
        grid_h = int(np.ceil((y_max - y_min) / grid_res)) + 1
        
        print(f"   -> Creating adjustment grid ({grid_w}x{grid_h}) resolution={grid_res}m")

        # グリッドの各セルに含まれる差分の平均を計算
        # Accumulate sums and counts
        grid_sum = np.zeros((grid_h, grid_w), dtype=np.float32)
        grid_count = np.zeros((grid_h, grid_w), dtype=np.float32)

        # 座標をインデックスに変換
        idx_x = ((g_pts[:, 0] - x_min) / grid_res).astype(np.int32)
        idx_y = ((y_max - g_pts[:, 1]) / grid_res).astype(np.int32) # Image coordinates (Top-down)

        # 範囲チェック
        valid_idx = (idx_x >= 0) & (idx_x < grid_w) & (idx_y >= 0) & (idx_y < grid_h)
        idx_x = idx_x[valid_idx]
        idx_y = idx_y[valid_idx]
        diffs = diffs[valid_idx]

        np.add.at(grid_sum, (idx_y, idx_x), diffs)
        np.add.at(grid_count, (idx_y, idx_x), 1)

        # 平均値の計算（データがないところはNaN）
        with np.errstate(divide='ignore', invalid='ignore'):
            adjustment_grid = grid_sum / grid_count

        # # 5. 穴埋め (Inpainting) - データがない場所を埋める
        # mask = np.isnan(adjustment_grid).astype(np.uint8) * 255
        # # NaNを0にしてOpenCVで扱えるようにする
        # adjustment_grid_fill = np.nan_to_num(adjustment_grid, nan=0.0).astype(np.float32)
        
        # # OpenCVのInpaintは8bitか16bit画像が必要だが、ここでは近似的に
        # # navier-stokes等でなく、単純なモルフォロジーやresizeによる穴埋めを行う
        # # 簡易的に、有効な値の平均で埋める（あるいはkNN）
        # # ここでは「Nearest」で粗く埋めた後、Blurする
        
        # 有効な値のインデックス
        y_valid, x_valid = np.where(grid_count > 0)
        if len(y_valid) > 0:
            # SciPyのNearestNDInterpolatorで穴埋め
            points_valid = np.column_stack((x_valid, y_valid))
            vals_valid = adjustment_grid[y_valid, x_valid]
            interpolator = NearestNDInterpolator(points_valid, vals_valid)
            
            # 全グリッド座標
            Y, X = np.mgrid[0:grid_h, 0:grid_w]
            adjustment_grid_filled = interpolator((X, Y))
        else:
            adjustment_grid_filled = np.zeros_like(adjustment_grid)

        # 6. 【重要】スムージング (Gaussian Blur)
        # これが「がたがたにならない」ための肝。
        # カーネルサイズを大きくすると、より緩やかになる
        k_size = smooth_kernel_size | 1 # 奇数にする
        print(f"   -> Smoothing adjustment grid (Kernel: {k_size}x{k_size})...")
        smooth_adjustment = cv2.GaussianBlur(adjustment_grid_filled, (k_size, k_size), 0)

        # 7. 全点への適用 (Bilinear Interpolation)
        print("   -> Applying smooth Z adjustment to all points...")
        
        # RegularGridInterpolatorの準備
        # grid座標系: x=0..w, y=0..h (yは画像座標なので上から下)
        # adjustment_gridは [y, x]
        # x_coords = np.arange(grid_w) * grid_res + x_min + grid_res/2
        # y_coords = y_max - (np.arange(grid_h) * grid_res + grid_res/2)
        
        # 画像座標系(y=0がTop=MaxY)に合わせてInterpolateする
        y_axis = np.arange(grid_h)
        x_axis = np.arange(grid_w)
        interp_func = RegularGridInterpolator((y_axis, x_axis), smooth_adjustment, bounds_error=False, fill_value=None)

        # LASの全点のグリッド座標を計算
        all_x_idx = (las_data.x - x_min) / grid_res
        all_y_idx = (y_max - las_data.y) / grid_res
        
        # 補正値を取得
        # RegularGridInterpolator takes (y, x)
        pts_for_interp = np.column_stack((all_y_idx, all_x_idx))
        z_offsets = interp_func(pts_for_interp)
        
        # 適用
        las_data.z += z_offsets
        
        print("   -> Z-alignment complete.")
        return

    # -------------------------------------------------------------------------
    # 8. ファイル保存
    # -------------------------------------------------------------------------
    def save_las(self, las_data, output_path, target_crs):
        print(f":: Saving LAS file -> {output_path}")
        
        new_header = laspy.LasHeader(point_format=las_data.header.point_format, version=las_data.header.version)
        new_header.scales = np.array([0.001, 0.001, 0.001])
        new_header.offsets = np.array([np.min(las_data.x), np.min(las_data.y), np.min(las_data.z)])

        try:
            new_header.add_crs(target_crs)
        except Exception:
            pass

        new_las = laspy.LasData(new_header)
        new_las.x = las_data.x
        new_las.y = las_data.y
        new_las.z = las_data.z

        for dim_name in las_data.point_format.dimension_names:
            if dim_name not in ['X', 'Y', 'Z']:
                try:
                    new_las[dim_name] = las_data[dim_name]
                except Exception:
                    pass

        new_las.write(output_path)
        print(":: Done.")

    def save_result_overlay(self, img_las, img_shp, M, output_path="result_overlay.png"):
        h, w = img_shp.shape
        img_las_transformed = cv2.warpAffine(img_las, M, (w, h))
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        vis_img[:, :, 2] = img_shp             
        vis_img[:, :, 1] = img_las_transformed 
        cv2.imwrite(output_path, vis_img)

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
def georeference_main():
    parser = argparse.ArgumentParser(description="Auto Georeferencing LAS to SHP + DEM (Smooth)")
    parser.add_argument("las_path", help="Input LAS file path")
    parser.add_argument("shp_path", help="Reference Road Edge SHP file path")
    parser.add_argument("dem_path",  help="Reference DEM GeoTIFF path")
    parser.add_argument("--las_epsg", type=int, default="32653", help="Original EPSG code of LAS") #Scaniverseで取得した点群は EPSG:32653 (UTM53N)
    parser.add_argument("--target_epsg", type=int, default=None, help="Force Target EPSG code")
    parser.add_argument("--pixel_size", type=float, default=0.1, help="Pixel size in meters")
    parser.add_argument("--smooth_kernel", type=int, default=51, help="Kernel size for Z smoothing (odd number)")
    
    args = parser.parse_args()

    myt_delta = datetime.timedelta(hours=8)
    MYT = datetime.timezone(myt_delta, 'MYT')
    timestamp = datetime.datetime.now(MYT).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"G_{timestamp}")
    os.makedirs(output_dir, exist_ok = True)
    print(f"::Output directory {os.path.abspath(output_dir)}")

    geo = AutoGeoreferencer(pixel_size=args.pixel_size)

    # 1. Load LAS
    las_data, pcd = geo.load_las(args.las_path)
    if las_data is None: return

    # 2. Load SHP
    print(f":: Reading SHP file: {args.shp_path}")
    try:
        gdf = gpd.read_file(args.shp_path)
    except Exception as e:
        print(f"!! Error reading SHP file: {e}")
        return

    target_crs = None
    if args.target_epsg is not None:
        target_crs = CRS.from_epsg(args.target_epsg)
        gdf.set_crs(target_crs, allow_override=True, inplace=True)
    elif gdf.crs is not None:
        target_crs = gdf.crs
        print(f"   -> Target CRS detected: {target_crs.name}")
    else:
        print("!! ERROR: No CRS detected. Use --target_epsg.")
        return

    # 3. Convert CRS (XY)
    las_data, pcd = geo.convert_las_crs(las_data, pcd, src_epsg=args.las_epsg, target_crs=target_crs)
    if las_data is None: return

    # 4. Grid Setup
    bbox = geo.setup_grid(las_data, buffer=20.0)

    # 5. Process Images
    img_shp = geo.process_shp(gdf, bbox)
    img_las = geo.process_las_slice(pcd, h_min=1.0, h_max=1.5)
    
    if img_shp is None or img_las is None: 
        print("!! Error generating process images.")
        return

    cv2.imwrite(os.path.join(output_dir, "debug_shp.png"), img_shp)
    cv2.imwrite(os.path.join(output_dir, "debug_las.png"), img_las)

    # 6. Calc Transform
    M_pixel = geo.calculate_transform(img_las, img_shp)
    
    if M_pixel is not None:
        geo.save_result_overlay(img_las, img_shp, M_pixel, output_path=os.path.join(output_dir, "result_overlay.png"))
        
        # Apply XY Transform
        las_data = geo.apply_xy_transform_memory(las_data, M_pixel)
        
        # 7. Apply Smooth Z Transform
        if args.dem_path:
            geo.align_z_smoothly(las_data, args.dem_path, target_crs, smooth_kernel_size=args.smooth_kernel)
        
        # Save
        base_name = os.path.splitext(os.path.basename(args.las_path))[0]
        output_filename = f"{base_name}_AG.las"
        final_output_path = os.path.join(output_dir, output_filename)
        geo.save_las(las_data, final_output_path, target_crs)

        # Log
        dx_m = M_pixel[0, 2] * args.pixel_size
        dy_m = M_pixel[1, 2] * args.pixel_size
        rotation_rad = np.arctan2(M_pixel[1, 0], M_pixel[0, 0])
        
        with open(os.path.join(output_dir, "processing_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"XY Correction X: {dx_m:.4f} m\n")
            f.write(f"XY Correction Y: {dy_m:.4f} m\n")
            f.write(f"Rotation: {np.degrees(rotation_rad):.4f} degrees\n")
            if args.dem_path:
                f.write(f"Used DEM: {args.dem_path} (Smooth Local Adjustment)\n")
    else:
        print("!! Alignment failed.")

if __name__ == "__main__":
    georeference_main()