"""
Automatic point cloud registration
"""

import argparse
import open3d as o3d
import numpy as np
import laspy as lp
import cv2
import os
import re
import CSF
from scipy.spatial import cKDTree
import copy
import datetime
import time
from functools import wraps
import platform
import psutil
import scipy

time_log = []

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f":: {func.__name__}")
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() -t0
        print(f"   -> Completed in {elapsed: 4f}s")
        time_log.append((func.__name__, elapsed))
        return result
    return wrapper


class AutoRegistrator:
    """Executes automatic point cloud registration."""
    def __init__(self, voxel_size, resolution):
        self.voxel_size = voxel_size
        self.scale = 1.0 / resolution
        self.min_xy = None
        self.max_xy = None
        self.img_w = 0
        self.img_h = 0

    @timer
    def load_las(self, input_path, keep_raw):
        print(f":: Loading LAS file: {input_path}")

        try:
            las = lp.read(input_path)
        except Exception as e:
            print(f"!! Error reading LAS: {e}")
            return None, None

        points = np.vstack((las.x, las.y, las.z)).T

        if hasattr(las, 'red'):
            colors = np.stack([las.red, las.green, las.blue], axis=1)
            max_val = 65535.0 if colors.max() > 255 else 255.0
            colors = np.clip(colors / max_val, 0.0, 1.0)
        else:
            colors = np.ones((points.shape[0], 3))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if self.voxel_size > 0 and not keep_raw:
            pcd = pcd.voxel_down_sample(self.voxel_size)
        print(f"   -> Loaded {len(pcd.points)} points (downsampled with voxel={self.voxel_size}m)")
        return pcd, las

    @timer
    def extract_overlap(self, src_pcd, tgt_pcd, threshold):
        print(f":: Extracting overlap region (XY threshold: {threshold}m)")

        src_pts = np.asarray(src_pcd.points)
        tgt_pts = np.asarray(tgt_pcd.points)

        if len(src_pts) == 0 or len(tgt_pts) == 0:
            return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

        tgt_tree = cKDTree(tgt_pts[:, :2])
        src_dists, _ = tgt_tree.query(src_pts[:, :2], k=1, workers=-1)
        src_mask = src_dists < threshold

        src_tree = cKDTree(src_pts[:, :2])
        tgt_dists, _ = src_tree.query(tgt_pts[:, :2], k=1, workers=-1)
        tgt_mask = tgt_dists < threshold
        print(f"   -> Src: {np.sum(src_mask)} / {len(src_pts)} pts | Tgt: {np.sum(tgt_mask)} / {len(tgt_pts)} pts")
        
        return src_pcd.select_by_index(np.where(src_mask)[0]), tgt_pcd.select_by_index(np.where(tgt_mask)[0])

    @timer
    def filter_ground(self, pcd, cloth_resolution, rigidness, class_threshold):
        print(":: Filtering ground with CSF")

        csf = CSF.CSF()
        csf.params.cloth_resolution = cloth_resolution
        csf.params.rigidness = rigidness
        csf.params.class_threshold = class_threshold

        points = np.asarray(pcd.points, dtype=np.float64)
        csf.setPointCloud(points)

        ground_indices = CSF.VecInt()
        non_ground_indices = CSF.VecInt()
        csf.do_filtering(ground_indices, non_ground_indices)

        ground = pcd.select_by_index(list(ground_indices))
        non_ground = pcd.select_by_index(list(non_ground_indices))
        print(f"   -> Ground: {len(ground.points)} pts | Non-ground: {len(non_ground.points)} pts")
        
        return ground, non_ground

    @timer
    def extract_slice(self, non_ground, ground, h_min, h_max):
        pts_ng = np.asarray(non_ground.points)
        pts_g = np.asarray(ground.points)

        if len(pts_g) == 0 or len(pts_ng) == 0:
            return o3d.geometry.PointCloud()

        tree = cKDTree(pts_g[:, :2])
        _, idx = tree.query(pts_ng[:, :2], k=1)
        ground_z = pts_g[idx, 2]

        diff = pts_ng[:, 2] - ground_z
        mask = (diff >= h_min) & (diff < h_max)

        return non_ground.select_by_index(np.where(mask)[0])

    @timer
    def project_to_image(self, src_pcd, tgt_pcd):
        src_pts = np.asarray(src_pcd.points)
        tgt_pts = np.asarray(tgt_pcd.points)

        if self.min_xy is None:
            if len(src_pts) > 0 and len(tgt_pts) > 0:
                pts_all = np.vstack([src_pts[:, :2], tgt_pts[:, :2]])
            else:
                return None, None
            if len(pts_all) > 0:
                self.min_xy = pts_all.min(axis=0)
                self.max_xy = pts_all.max(axis=0)
                width_m = self.max_xy[0] - self.min_xy[0]
                height_m = self.max_xy[1] - self.min_xy[1]
                self.img_w = int(np.ceil(width_m * self.scale)) + 10
                self.img_h = int(np.ceil(height_m * self.scale)) + 10
                print(f":: Projecting point cloud to 2D image")

        def project(pts):
            if len(pts) == 0:
                return np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            px = np.round(((pts[:, 0] - self.min_xy[0]) * self.scale)).astype(np.int32)
            py = np.round(((self.max_xy[1] - pts[:, 1]) * self.scale)).astype(np.int32)
            pts_norm = np.column_stack([px, py])
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            valid = (
                (pts_norm[:, 0] >= 0) & (pts_norm[:, 0] < self.img_w) &
                (pts_norm[:, 1] >= 0) & (pts_norm[:, 1] < self.img_h)
            )
            pv = pts_norm[valid]
            img[pv[:, 1], pv[:, 0]] = 255
            return img
        return project(src_pts), project(tgt_pts)

    def estimate_similarity(self, src_pts, tgt_pts):
        src_mean = np.mean(src_pts, axis=0)
        tgt_mean = np.mean(tgt_pts, axis=0)
        src_c = src_pts - src_mean
        tgt_c = tgt_pts - tgt_mean

        U, _, Vt = np.linalg.svd(src_c.T @ tgt_c)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - R @ src_mean
        return np.hstack([R, t.reshape(2, 1)])

    @timer
    def register_with_ORB(self, src_img, tgt_img, orb_nfeatures, ransac_iter, ransac_threshold, lowe_ratio):
        if np.sum(src_img) == 0 or np.sum(tgt_img) == 0:
            return np.eye(3), 0

        orb = cv2.ORB_create(orb_nfeatures)
        src_kp, src_des = orb.detectAndCompute(src_img, None)
        tgt_kp, tgt_des = orb.detectAndCompute(tgt_img, None)

        if src_des is None or tgt_des is None or len(src_des) < 5 or len(tgt_des) < 5:
            return np.eye(3), 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(src_des, tgt_des, k=2)

        good = []
        for m, n in matches:
            if m.distance < lowe_ratio * n.distance:
                good.append(m)

        if len(good) < 5:
            return np.eye(3), 0

        src_pts = np.float32([src_kp[m.queryIdx].pt for m in good])
        tgt_pts = np.float32([tgt_kp[m.trainIdx].pt for m in good])

        best_inliers_count = 0
        best_M = np.eye(3)
        N = len(src_pts)

        for _ in range(ransac_iter):
            if N < 2:
                break
            idx = np.random.choice(N, 2, replace=False)
            M_cand_2x3 = self.estimate_similarity(src_pts[idx], tgt_pts[idx])
            src_homo = np.hstack([src_pts, np.ones((N, 1))])
            transformed = (M_cand_2x3 @ src_homo.T).T
            errors = np.linalg.norm(transformed - tgt_pts, axis=1)
            inliers = np.sum(errors < ransac_threshold)
            if inliers > best_inliers_count:
                best_inliers_count = inliers
                best_M = np.eye(3)
                best_M[:2, :] = M_cand_2x3

        return best_M, best_inliers_count

    def point2point_rmse(self, src_nonground, tgt_nonground, M_pixel, k=6):
            if src_nonground.is_empty() or tgt_nonground.is_empty():
                return float('inf')

            src_transformed, _ = self.apply_transform(src_nonground, M_pixel)

            src_pts = np.asarray(src_transformed.points)[:,:2]
            tgt_pts = np.asarray(tgt_nonground.points)[:,:2]

            if len(src_pts) == 0 or len(tgt_pts) < k:
                return float('inf')

            tgt_tree = cKDTree(tgt_pts)
            dists, _ = tgt_tree.query(src_pts, k=1, workers=-1)
            rmse = np.sqrt(np.mean(dists ** 2))

            return rmse

    @timer
    def apply_transform(self, pcd, M_pixel):
        if pcd.is_empty():
            return pcd, np.eye(4)

        s = self.scale
        mx, my = self.min_xy
        H = self.img_h

        # World-to-image transform (T_w2i)
        T_w2i = np.eye(3)
        T_w2i[0, 0] = s
        T_w2i[0, 2] = -s * mx
        T_w2i[1, 1] = -s
        T_w2i[1, 2] = (H - 1.0) + s * my

        try:
            T_i2w = np.linalg.inv(T_w2i)
        except np.linalg.LinAlgError:
            return pcd, np.eye(4)

        # Convert pixel-space transform to world-space
        M_world_2d = T_i2w @ M_pixel @ T_w2i

        M_world_3d = np.eye(4)
        M_world_3d[:2, :2] = M_world_2d[:2, :2]
        M_world_3d[:2, 3] = M_world_2d[:2, 2]

        new_pcd = copy.deepcopy(pcd)
        new_pcd.transform(M_world_3d)

        return new_pcd, M_world_3d

    @timer
    def register_with_2DICP(self, src_pcd, tgt_pcd, threshold, max_iteration, relative_fitness, relative_rmse):
        print(f":: Running 2D-ICP refinement (threshold: {threshold}m)")

        src = copy.deepcopy(src_pcd)
        tgt = copy.deepcopy(tgt_pcd)

        src_pts = np.asarray(src.points)
        tgt_pts = np.asarray(tgt.points)

        if len(src_pts) == 0 or len(tgt_pts) == 0:
            return np.eye(4)

        src_pts[:, 2] = 0
        tgt_pts[:, 2] = 0
        src.points = o3d.utility.Vector3dVector(src_pts)
        tgt.points = o3d.utility.Vector3dVector(tgt_pts)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=relative_fitness,
            relative_rmse=relative_rmse,
            max_iteration=max_iteration
        )

        reg_p2p = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria
        )

        print(f"   -> Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}m")
        return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

    @timer
    def estimate_tilt_shift(self, src_ground_aligned, tgt_ground):
        src_pts = np.asarray(src_ground_aligned.points)
        tgt_pts = np.asarray(tgt_ground.points)

        if len(src_pts) < 10 or len(tgt_pts) < 10:
            print("!! Not enough ground points for Z correction")
            return 0.0, 0.0, 0.0

        tree = cKDTree(tgt_pts[:, :2])
        dists, indices = tree.query(src_pts[:, :2], k=1, distance_upper_bound=0.5)

        valid = dists != float('inf')
        if np.sum(valid) < 10:
            print("!! Too few overlapping ground points; falling back to mean Z shift")
            z_diff = np.mean(tgt_pts[:, 2]) - np.mean(src_pts[:, 2])
            return 0.0, 0.0, z_diff

        x_s = src_pts[valid, 0]
        y_s = src_pts[valid, 1]
        z_s = src_pts[valid, 2]
        z_t = tgt_pts[indices[valid], 2]

        diff_z = z_t - z_s
        A = np.vstack([x_s, y_s, np.ones(len(x_s))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, diff_z, rcond=None)
        a, b, c = coeffs

        print(f"   -> Tilt X: {a:.6f}, Tilt Y: {b:.6f}, Z shift: {c:.4f}m")
        return a, b, c

    @timer
    def save_las(self, output_path, original_las, aligned_points):
        print(f":: Saving registered point cloud as LAS: {output_path}")
        try:
            min_coords = np.min(aligned_points, axis=0)

            new_header = lp.LasHeader(point_format=original_las.header.point_format, version=original_las.header.version)
            new_header.offsets = min_coords
            new_header.scales = original_las.header.scales

            new_las = lp.LasData(header=new_header)
            new_las.x = aligned_points[:, 0]
            new_las.y = aligned_points[:, 1]
            new_las.z = aligned_points[:, 2]

            for dim in ['red', 'green', 'blue', 'intensity', 'classification']:
                try:
                    setattr(new_las, dim, getattr(original_las, dim))
                except Exception:
                    pass

            new_las.write(output_path)
            print(f"   -> Saved {len(aligned_points)} points successfully")
        except Exception as e:
            print(f"!! Error saving LAS: {e}")


class RegistrationExporter:
    """Handles overlay image and log output for a registration run."""
    def __init__(self, log_dir, base_filename, resolution):
        self.log_dir = log_dir
        self.base_filename = base_filename
        self.scale = 1 / resolution
        self.kernel = np.ones((3, 3), np.uint8)        
        self.ground_bounds = None
        self.side_w = None
        self.side_h = None

    def save_slice_overlay(self, src_slice, tgt_slice, global_reg, suffix):
        src_img, tgt_img = global_reg.project_to_image(src_slice, tgt_slice)
        h_img, w_img = tgt_img.shape
        
        overlay_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        overlay_img[..., 1] = src_img
        overlay_img[..., 2] = tgt_img
        
        img_output_path = os.path.join(self.log_dir, f"{self.base_filename}_{suffix}_overlay.png")
        cv2.imwrite(img_output_path, overlay_img)
        print(f"   -> Saved: {img_output_path}")

    def init_ground_bounds(self, tgt_ground, src_ground_init):
        if self.ground_bounds is None:
            tgt_pts = np.asarray(tgt_ground.points)
            src_pts = np.asarray(src_ground_init.points)
            
            all_x = np.concatenate([tgt_pts[:, 0], src_pts[:, 0]])
            all_z = np.concatenate([tgt_pts[:, 2], src_pts[:, 2]])
            
            raw_min_x, raw_max_x = all_x.min(), all_x.max()
            raw_min_z, raw_max_z = all_z.min(), all_z.max()
            
            min_x, max_x = raw_min_x - 10, raw_max_x + 10
            min_z, max_z = raw_min_z - 10, raw_max_z + 10
            
            range_x = max_x - min_x
            range_z = max_z - min_z
            
            self.side_w = int(np.ceil(range_x * self.scale))
            self.side_h = int(np.ceil(range_z * self.scale))
            
            self.ground_bounds = (min_x, min_z)

    def project_to_side_view(self, pts):
        img = np.zeros((self.side_h, self.side_w), dtype=np.uint8)
        if len(pts) == 0 or self.ground_bounds is None:
            return img
            
        min_x, min_z = self.ground_bounds
        
        x_px = ((pts[:, 0] - min_x) * self.scale).astype(np.int32)
        z_px = ((pts[:, 2] - min_z) * self.scale).astype(np.int32)
        
        valid = (x_px >= 0) & (x_px < self.side_w) & (z_px >= 0) & (z_px < self.side_h)
        
        img[self.side_h - z_px[valid] - 1, x_px[valid]] = 255
        return cv2.dilate(img, self.kernel, iterations=1)

    def save_ground_side_overlay(self, src_ground, tgt_ground, suffix):
        self.init_ground_bounds(tgt_ground, src_ground)
        
        tgt_pts = np.asarray(tgt_ground.points)
        src_pts = np.asarray(src_ground.points)
        
        tgt_img = self.project_to_side_view(tgt_pts)
        src_img = self.project_to_side_view(src_pts)
        
        overlay_img = np.zeros((self.side_h, self.side_w, 3), dtype=np.uint8)
        overlay_img[..., 1] = src_img
        overlay_img[..., 2] = tgt_img
        
        img_output_path = os.path.join(self.log_dir, f"{self.base_filename}_{suffix}_overlay.png")
        cv2.imwrite(img_output_path, overlay_img)
        print(f"   -> Saved: {img_output_path}")

    def write_execution_log(self, args, is_success, timestamp, process_time, best_height, best_score, icp_fitness, icp_rmse, tilt_params, T_final, ram_before, ram_after, peak_ram, cpu_avg):
        log_path = os.path.join(self.log_dir, f"{self.base_filename}_log.txt")
        matrix_path = os.path.join(self.log_dir, f"{self.base_filename}_matrix.txt")
        print(f":: Writing execution log: {log_path}")
 
        with open(log_path, "w") as f:
            f.write("=== Registration Log ===\n")
            f.write(f"Timestamp     : {timestamp}\n")
            f.write(f"Process Time  : {process_time}\n")
            f.write(f"Source        : {args.source}\n")
            f.write(f"Target        : {args.target}\n")
            f.write(f"Status        : {'SUCCESS' if is_success else 'FAILED'}\n\n")

            f.write("--- Environment ---\n")
            f.write(f"OS                : {platform.system()} {platform.release()} ({platform.version()})\n")
            f.write(f"Python            : {platform.python_version()}\n")
            f.write(f"CPU               : {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count(logical=True)} logical)\n")
            f.write(f"CPU model         : {platform.processor()}\n")
            f.write(f"CPU usage (avg)   : {cpu_avg:.1f}%\n")
            f.write(f"RAM total         : {psutil.virtual_memory().total / 1024**3:.1f} GB\n")
            f.write(f"RAM before        : {ram_before:.2f} GB\n")
            f.write(f"RAM after         : {ram_after:.2f} GB\n")
            f.write(f"RAM delta         : {ram_after - ram_before:+.2f} GB\n")
            f.write(f"RAM peak (proc)   : {peak_ram:.2f} GB\n\n")            

            f.write("--- Library Versions ---\n")
            f.write(f"open3d            : {o3d.__version__}\n")
            f.write(f"numpy             : {np.__version__}\n")
            f.write(f"opencv            : {cv2.__version__}\n")
            f.write(f"laspy             : {lp.__version__}\n")
            f.write(f"scipy             : {scipy.__version__}\n\n")

            f.write("--- Parameters ---\n")
            for key, val in vars(args).items():
                f.write(f"{key:<25}: {val}\n")
            print("\n")
            
            if is_success:
                f.write("\n--- Results ---\n")
                f.write(f"Best height slice : {best_height:.1f}m\n")
                f.write(f"Point2Point RMSE  : {best_score:.4f}m\n")
                f.write(f"ICP fitness       : {icp_fitness:.2f}\n")
                f.write(f"ICP RMSE          : {icp_rmse:.4f}m\n")
                if tilt_params:
                    a, b, c = tilt_params
                    f.write(f"Tilt parameters   : a={a:.6f}, b={b:.6f}, c={c:.4f}\n")
                f.write(f"Matrix file       : {os.path.basename(matrix_path)}\n")
                
                f.write("\n--- time ---\n")
                for label, elapsed in time_log:
                    f.write(f"{label:<30}: {elapsed:.4f}s\n")
                f.write(f"{'Total':<30}: {process_time}s\n")      
        print(f"   -> Saved: {log_path}")

        if is_success and T_final is not None:
            np.savetxt(matrix_path, T_final, fmt="%.10f", delimiter=",")
            print(f"   -> Saved: {matrix_path}")
            

def registration_main():
    parser = argparse.ArgumentParser(description="ORB-based LAS point cloud registration with ICP refinement")

    # --- Input / Output ---
    parser.add_argument("source", help="Source LAS file path")
    parser.add_argument("target", help="Target LAS file path")
    parser.add_argument("--out_dir", type=str, default=None, help="Output subdirectory inside results/")

    # --- Preprocessing ---
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel downsampling size in meters (default: 0.005)")
    parser.add_argument("--pixel_size", type=float, default=0.045, help="Projection resolution in meters/pixel (default: 0.045)")
    parser.add_argument("--overlap_threshold", type=float, default=1.0, help="XY distance threshold for overlap extraction in meters (default: 1.0)")

    # --- CSF-based ground filtering ---
    parser.add_argument("--csf_cloth_res", type=float, default=1.0, help="CSF cloth resolution (default: 1.0)")
    parser.add_argument("--csf_rigidness", type=int, default=3, help="CSF rigidness parameter (default: 3)")
    parser.add_argument("--csf_class_threshold", type=float, default=0.5, help="CSF classification threshold (default: 0.5)")

    # --- Height slice search ---
    parser.add_argument("--slice_h_min", type=float, default=0.7, help="Minimum height of slice search range in meters (default: 0.7)")
    parser.add_argument("--slice_h_max", type=float, default=2.5, help="Maximum height of slice search range in meters (default: 2.5)")
    parser.add_argument("--slice_step", type=float, default=0.2, help="Step size for height slice search in meters (default: 0.2)")
    parser.add_argument("--slice_thickness", type=float, default=0.2, help="Thickness of each height slice in meters (default: 0.2)")

    # --- ORB-based registration ---
    parser.add_argument("--orb_nfeatures", type=int, default=5000, help="Number of ORB features to detect (default: 5000)")
    parser.add_argument("--ransac_iter", type=int, default=30000, help="Number of RANSAC iterations (default: 10000)")
    parser.add_argument("--ransac_threshold", type=float, default=3.0, help="RANSAC inlier threshold in pixels (default: 3.0)")
    parser.add_argument("--min_inliers", type=int, default=10, help="Minimum inlier count to accept an ORB match (default: 10)")
    parser.add_argument("--lowe_ratio", type=float, default=0.75, help="Lowe's ratio test threshold for ORB matching (default: 0.75)")

    # --- 2DICP-based registration ---
    parser.add_argument("--icp_threshold_factor", type=float, default=0.75, help="ICP threshold = pixel_size * this factor (default: 0.75)")
    parser.add_argument("--icp_max_iter", type=int, default=2000, help="Maximum ICP iterations (default: 2000)")
    parser.add_argument("--icp_relative_fitness", type=float, default=1e-6, help="ICP convergence criterion: relative fitness (default: 1e-6)")
    parser.add_argument("--icp_relative_rmse", type=float, default=1e-6, help="ICP convergence criterion: relative RMSE (default: 1e-6)")
    
    # --- Log ---
    parser.add_argument("--no_log", action="store_true", help="Suppress all intermediate and final image/log outputs (default: outputs enabled)")
    
    args = parser.parse_args()

    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    ram_before = psutil.virtual_memory().used / 1024**3
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    start_time = datetime.datetime.now()

    output_dir = os.path.join(f"results", args.out_dir) if args.out_dir else f"results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    max_index = 0
    for entry in os.listdir(output_dir):
        match = re.match(r'^r(\d+)', entry)
        if match:
            try:
                idx = int(match.group(1))
                if idx > max_index:
                    max_index = idx
            except ValueError:
                pass

    new_index = max_index + 1
    log_dir = os.path.join(output_dir, f"r{new_index}_log")
    os.makedirs(log_dir, exist_ok=True)
    output_las_path = os.path.join(output_dir, f"r{new_index}.las")
    base_filename = f"r{new_index}"
    print(f":: Output directory: {os.path.abspath(output_dir)}")

    registrator = AutoRegistrator(args.voxel_size, resolution=args.pixel_size)

    src_pcd, SRC_las = registrator.load_las(args.source, keep_raw=False)
    tgt_pcd, _ = registrator.load_las(args.target, keep_raw=False)

    src_pcd_ov, tgt_pcd_ov = registrator.extract_overlap(src_pcd, tgt_pcd, threshold=args.overlap_threshold)

    src_ground, src_nonground = registrator.filter_ground(
        src_pcd_ov,
        cloth_resolution=args.csf_cloth_res,
        rigidness=args.csf_rigidness,
        class_threshold=args.csf_class_threshold
    )
    tgt_ground, tgt_nonground = registrator.filter_ground(
        tgt_pcd_ov,
        cloth_resolution=args.csf_cloth_res,
        rigidness=args.csf_rigidness,
        class_threshold=args.csf_class_threshold
    )

    best_score = float('inf')
    best_M = np.eye(3)
    best_height = 0.0

    search_ranges = np.arange(args.slice_h_min, args.slice_h_max, args.slice_step)
    print(f":: Searching best height slice for ORB matching ({len(search_ranges)} steps)")

    for h in search_ranges:
        src_slice = registrator.extract_slice(src_nonground, src_ground, h, h + args.slice_thickness)
        tgt_slice = registrator.extract_slice(tgt_nonground, tgt_ground, h, h + args.slice_thickness)

        src_img, tgt_img = registrator.project_to_image(src_slice, tgt_slice)

        M_cand, inliers = registrator.register_with_ORB(
            src_img, tgt_img,
            orb_nfeatures=args.orb_nfeatures,
            ransac_iter=args.ransac_iter,
            ransac_threshold=args.ransac_threshold,
            lowe_ratio=args.lowe_ratio
        )

        if inliers > args.min_inliers:
            score = registrator.point2point_rmse(src_nonground, tgt_nonground, M_cand)
            print(f"   H={h:.1f}m: rmse={score:.4f}m, inliers={inliers}")
            if score < best_score:
                best_score = score
                best_M = M_cand
                best_height = h
        else:
            print(f"   H={h:.1f}m: not enough matches ({inliers}), skipping")

    is_success = best_score != float('inf')

    if not is_success:
        print("\n!! Registration FAILED: no valid ORB matches found")
        T_final = np.eye(4)
    else:
        print(f"\n   -> Best slice: H={best_height:.1f}m, rmse={best_score:.4f}m")

        print(":: Applying ORB transform to full-resolution cloud")
        SRC_pts = np.vstack((SRC_las.x, SRC_las.y, SRC_las.z)).T
        SRC_pcd = o3d.geometry.PointCloud()
        SRC_pcd.points = o3d.utility.Vector3dVector(SRC_pts)

        SRC_aligned, T_orb = registrator.apply_transform(SRC_pcd, best_M)
        src_ground_aligned, _ = registrator.apply_transform(src_ground, best_M)
        src_nonground_aligned, _ = registrator.apply_transform(src_nonground, best_M)
        print(f"   -> ORB transform applied to {len(SRC_pts)} points")

        src_slice_aligned = registrator.extract_slice(
            src_nonground_aligned, src_ground_aligned,
            best_height, best_height + args.slice_thickness
        )
        tgt_slice_ref = registrator.extract_slice(
            tgt_nonground, tgt_ground,
            best_height, best_height + args.slice_thickness
        )

        icp_threshold = args.pixel_size * args.icp_threshold_factor
        M_icp_4x4, icp_fitness, icp_rmse = registrator.register_with_2DICP(
            src_slice_aligned, tgt_slice_ref,
            threshold=icp_threshold,
            max_iteration=args.icp_max_iter,
            relative_fitness=args.icp_relative_fitness,
            relative_rmse=args.icp_relative_rmse
        )
        SRC_aligned.transform(M_icp_4x4)
        src_ground_aligned.transform(M_icp_4x4)

        print(":: Estimating Z tilt and shift from ground planes")
        a, b, c = registrator.estimate_tilt_shift(src_ground_aligned, tgt_ground)

        T_z = np.eye(4)
        T_z[2, 0] = a
        T_z[2, 1] = b
        T_z[2, 3] = c

        SRC_pts = np.asarray(SRC_aligned.points)
        SRC_pts[:, 2] += (a * SRC_pts[:, 0]) + (b * SRC_pts[:, 1]) + c

        registrator.save_las(output_las_path, SRC_las, SRC_pts)

        process_time = datetime.datetime.now() - start_time
        print(f":: Total processing time: {process_time}s")
        ram_after = psutil.virtual_memory().used / 1024**3
        peak_ram = proc.memory_info().rss / 1024**3
        cpu_avg = proc.cpu_percent(interval=1)

    if not args.no_log:
        print(":: Generating result overlay images and logs")
 
        exporter = RegistrationExporter(log_dir, base_filename, args.pixel_size)

        vis_height = best_height if is_success else 1.0
        src_ground_original = copy.deepcopy(src_ground)
        src_nonground_original = copy.deepcopy(src_nonground)
 
        src_slice_original = registrator.extract_slice(
            src_nonground_original, src_ground_original,
            vis_height, vis_height + args.slice_thickness
        )
        tgt_slice_final = registrator.extract_slice(
            tgt_nonground, tgt_ground,
            vis_height, vis_height + args.slice_thickness
        )

        exporter.save_slice_overlay(src_slice_original, tgt_slice_final, registrator, "01_original")
        
        src_slice_ORB = src_slice_original.transform(T_orb)
        exporter.save_slice_overlay(src_slice_ORB, tgt_slice_final, registrator, "02_ORB")
        
        src_slice_2DICP = src_slice_ORB.transform(M_icp_4x4)
        exporter.save_slice_overlay(src_slice_2DICP, tgt_slice_final, registrator, "03_ICP")

        exporter.save_ground_side_overlay(src_ground, tgt_ground, "04_original_ground")
        
        T_final = T_z @ M_icp_4x4 @ T_orb
        src_ground_final = copy.deepcopy(src_ground).transform(T_final)
        exporter.save_ground_side_overlay(src_ground_final, tgt_ground, "05_shifted_ground")

        tilt_params = (a, b, c) if is_success else None
        exporter.write_execution_log(
            args, is_success, timestamp, 
            process_time, best_height, 
            best_score, icp_fitness, 
            icp_rmse, tilt_params, 
            T_final, ram_before, 
            ram_after, peak_ram, cpu_avg)
               
if __name__ == "__main__":
    registration_main()