import argparse
import open3d as o3d
import numpy as np
import pandas as pd
import laspy as lp
import cv2
import os
import CSF
from scipy.spatial import cKDTree
import copy
import datetime

class GlobalRegistration:
    def __init__(self, voxel_size, resolution):
        """
        resolution: 1ピクセルあたりのメートル数 (例: 0.01 = 1cm)
        """
        self.voxel_size = voxel_size
        self.resolution = resolution
        # スケールは固定 (1m を何ピクセルにするか -> 1 / 0.01 = 100px)
        self.scale = 1.0 / self.resolution
        
        self.min_xy = None
        self.img_w = 0
        self.img_h = 0

    # -------------------------------------------------------------------------
    # ① LASデータの読み込み (座標系変換なし)
    # -------------------------------------------------------------------------
    def load_las_no_transform(self, input_path, keep_raw=False):
        print(f":: Loading {input_path}...")
        try:
            las = lp.read(input_path)
        except Exception as e:
            print(f"!! Error reading LAS: {e}")
            return None, None

        x = las.x
        y = las.y
        z = las.z
        points = np.vstack((x, y, z)).T

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
        
        return pcd, las

    # -------------------------------------------------------------------------
    # ② 地理的重複領域の抽出
    # -------------------------------------------------------------------------
    def extract_overlap_region(self, pcd1, pcd2, threshold=5.0):
        pts1 = np.asarray(pcd1.points)
        pts2 = np.asarray(pcd2.points)

        if len(pts1) == 0 or len(pts2) == 0:
            return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

        pts1_xy = pts1[:, :2]
        pts2_xy = pts2[:, :2]

        print(f":: Extracting Overlap (XY Threshold: {threshold}m)...")

        tree2 = cKDTree(pts2_xy)
        dists1, _ = tree2.query(pts1_xy, k=1, workers=-1)
        mask1 = dists1 < threshold

        tree1 = cKDTree(pts1_xy)
        dists2, _ = tree1.query(pts2_xy, k=1, workers=-1)
        mask2 = dists2 < threshold

        print(f"   Src overlap: {np.sum(mask1)} / {len(pts1)}")
        print(f"   Tgt overlap: {np.sum(mask2)} / {len(pts2)}")

        return pcd1.select_by_index(np.where(mask1)[0]), pcd2.select_by_index(np.where(mask2)[0])

    # -------------------------------------------------------------------------
    # ③ 点群のスライス化・画像化 (CSF使用)
    # -------------------------------------------------------------------------
    def filter_ground_csf(self, pcd):
        csf = CSF.CSF()
        csf.params.cloth_resolution = 1.0
        csf.params.rigidness = 3
        csf.params.class_threshold = 0.5
        
        points = np.asarray(pcd.points, dtype=np.float64)
        csf.setPointCloud(points)
        
        ground_indices = CSF.VecInt()
        non_ground_indices = CSF.VecInt()
        
        csf.do_filtering(ground_indices, non_ground_indices)
        
        ground = pcd.select_by_index(list(ground_indices))
        non_ground = pcd.select_by_index(list(non_ground_indices))
        
        return ground, non_ground

    def extract_slice(self, non_ground, ground, h_min, h_max):
        ng_pts = np.asarray(non_ground.points)
        g_pts = np.asarray(ground.points)
        
        if len(g_pts) == 0 or len(ng_pts) == 0:
            return o3d.geometry.PointCloud()

        tree = cKDTree(g_pts[:, :2])
        _, idx = tree.query(ng_pts[:, :2], k=1)
        ground_z = g_pts[idx, 2]
        
        diff = ng_pts[:, 2] - ground_z
        mask = (diff >= h_min) & (diff < h_max)
        
        sliced = non_ground.select_by_index(np.where(mask)[0])
        return sliced

    def pcd_to_image(self, pcd_src, pcd_tgt):
        pts_src = np.asarray(pcd_src.points)
        pts_tgt = np.asarray(pcd_tgt.points)
        
        # キャンバスサイズの決定（初回のみ、または未設定時）
        # 固定スケールに基づき、全点群が入る画像サイズを計算する
        if self.min_xy is None:
            pts_all = np.vstack([pts_src[:, :2], pts_tgt[:, :2]]) if len(pts_src)>0 and len(pts_tgt)>0 else pts_src[:,:2]
            
            if len(pts_all) > 0:
                # 最小座標を決定し、少しマージンを持たせる
                self.min_xy = pts_all.min(axis=0)
                max_xy = pts_all.max(axis=0)
                
                # ワールド座標での幅・高さ
                width_m = max_xy[0] - self.min_xy[0]
                height_m = max_xy[1] - self.min_xy[1]
                
                # ピクセル数に変換 (ceilで切り上げ)
                self.img_w = int(np.ceil(width_m * self.scale)) + 10  # +10px padding
                self.img_h = int(np.ceil(height_m * self.scale)) + 10
                
                print(f":: Image Canvas Size: {self.img_w} x {self.img_h} (Scale: {self.scale:.2f} px/m)")

        def project(pts):
            if len(pts) == 0: return np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            
            # 座標変換: (World - Min) * Scale
            pts_norm = (pts[:, :2] - self.min_xy) * self.scale
            pts_norm = np.round(pts_norm).astype(np.int32)
            
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            
            # 画像範囲内の点のみ描画
            valid_mask = (pts_norm[:, 0] >= 0) & (pts_norm[:, 0] < self.img_w) & \
                         (pts_norm[:, 1] >= 0) & (pts_norm[:, 1] < self.img_h)
            pts_valid = pts_norm[valid_mask]
            
            # 画像座標系へ (Y軸反転: 下から上へ伸びるWorld座標を、上から下のImage座標へ)
            # img[h - y - 1, x]
            img[self.img_h - pts_valid[:, 1] - 1, pts_valid[:, 0]] = 255
            return img

        return project(pts_src), project(pts_tgt)

    # -------------------------------------------------------------------------
    # ④ ORB特徴量による位置合わせ
    # -------------------------------------------------------------------------
    def estimate_similarity_fixed_scale(self, src_pts, tgt_pts):
        src_mean = np.mean(src_pts, axis=0)
        tgt_mean = np.mean(tgt_pts, axis=0)
        src_centered = src_pts - src_mean
        tgt_centered = tgt_pts - tgt_mean

        U, _, Vt = np.linalg.svd(src_centered.T @ tgt_centered)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1,:] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - R @ src_mean
        return np.hstack([R, t.reshape(2,1)])

    def align_rigid_orb(self, img_src, img_tgt):
        if np.sum(img_src) == 0 or np.sum(img_tgt) == 0:
            return np.eye(3), 0

        # 解像度が高いと特徴点が増えすぎる可能性があるため数を調整
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(img_src, None)
        kp2, des2 = orb.detectAndCompute(img_tgt, None)

        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return np.eye(3), 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 5:
            return np.eye(3), 0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        tgt_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        best_inliers_count = 0
        best_M = np.eye(3)
        
        n_iter = 10000
        # 1cm解像度の場合、許容誤差5px = 5cm程度
        threshold = 3.0 
        N = len(src_pts)
        
        for _ in range(n_iter):
            if N < 2: break
            idx = np.random.choice(N, 2, replace=False)
            M_cand_2x3 = self.estimate_similarity_fixed_scale(src_pts[idx], tgt_pts[idx])
            
            src_homo = np.hstack([src_pts, np.ones((N,1))])
            transformed = (M_cand_2x3 @ src_homo.T).T
            
            errors = np.linalg.norm(transformed - tgt_pts, axis=1)
            inliers = np.sum(errors < threshold)
            
            if inliers > best_inliers_count:
                best_inliers_count = inliers
                best_M = np.eye(3)
                best_M[:2, :] = M_cand_2x3

        return best_M, best_inliers_count

    def evaluate_mean(self, img_src, img_tgt, M):
        if np.sum(img_src) == 0 or np.sum(img_tgt) == 0: return float('inf')

        M_cv = M[:2, :] 
        src_y, src_x = np.where(img_src > 0)
        if len(src_x) == 0: return float('inf')
        src_pts = np.column_stack([src_x, src_y]).astype(np.float32).reshape(-1, 1, 2)
        src_trans = cv2.transform(src_pts, M_cv).reshape(-1, 2)

        tgt_y, tgt_x = np.where(img_tgt > 0)
        if len(tgt_x) == 0: return float('inf')
        tgt_pts = np.column_stack([tgt_x, tgt_y])
        
        tree = cKDTree(tgt_pts)
        dists, _ = tree.query(src_trans, k=1)
        return np.mean(dists)

    def apply_rigid_transform(self, pcd, M_pixel):
        if pcd.is_empty():
            return pcd

        s = self.scale
        mx, my = self.min_xy
        H = self.img_h  # 計算された高さを使用
        
        # Image -> World への変換マトリクス構築
        # Pixel (u, v) -> World (x, y)
        # u = (x - mx) * s  => x = u/s + mx
        # v = H - 1 - (y - my) * s => y = my + (H - 1 - v)/s
        
        # T_w2i (World to Image Matrix)
        T_w2i = np.eye(3)
        T_w2i[0, 0] = s
        T_w2i[0, 2] = -s * mx
        T_w2i[1, 1] = -s
        T_w2i[1, 2] = (H - 1.0) + s * my

        try:
            T_i2w = np.linalg.inv(T_w2i)
        except np.linalg.LinAlgError:
            return pcd

        # ピクセル空間での回転移動行列をワールド空間へ変換
        M_world_2d = T_i2w @ M_pixel @ T_w2i
        
        M_world_3d = np.eye(4)
        M_world_3d[:2, :2] = M_world_2d[:2, :2]
        M_world_3d[:2, 3]  = M_world_2d[:2, 2]

        # print(":: Applying restored rigid transformation (XY-only Matrix):")
        
        new_pcd = copy.deepcopy(pcd)
        new_pcd.transform(M_world_3d)
        
        return new_pcd

    # -------------------------------------------------------------------------
    # ④-B ICPによる微調整 (追加)
    # -------------------------------------------------------------------------
    def align_fine_icp(self, pcd_source, pcd_target, threshold):
        """
        ORB適用後の点群に対して、2D平面上でのICP微調整を行う。
        Z値を0に潰して計算することでXYのみの補正行列を求める。
        """
        print(f":: Running 2D-ICP Refinement (Threshold: {threshold}m)...")
        
        # オリジナルを保持するためコピー
        src = copy.deepcopy(pcd_source)
        tgt = copy.deepcopy(pcd_target)

        # 3D点群を2D平面に射影 (Z=0)
        np_src = np.asarray(src.points)
        np_tgt = np.asarray(tgt.points)
        
        if len(np_src) == 0 or len(np_tgt) == 0:
            return np.eye(4)

        np_src[:, 2] = 0
        np_tgt[:, 2] = 0
        
        src.points = o3d.utility.Vector3dVector(np_src)
        tgt.points = o3d.utility.Vector3dVector(np_tgt)
    
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,  # 改善がごくわずかでも継続
            relative_rmse=1e-6,     # 誤差減少がごくわずかでも継続
            max_iteration=2000      # 十分な回数を回す
        )

        # ICP実行 (Point-to-Point)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria
        )
        
        print(f"   ICP Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")
        return reg_p2p.transformation

    # -------------------------------------------------------------------------
    # ⑤ Z軸の傾きと高さの最適化
    # -------------------------------------------------------------------------
    def calculate_tilt_and_shift(self, s_ground_aligned, t_ground):
        pts_s = np.asarray(s_ground_aligned.points)
        pts_t = np.asarray(t_ground.points)

        if len(pts_s) < 10 or len(pts_t) < 10:
            print("!! Not enough ground points for tilt optimization.")
            return 0.0, 0.0, 0.0

        tree = cKDTree(pts_t[:, :2])
        dists, indices = tree.query(pts_s[:, :2], k=1, distance_upper_bound=0.5)

        valid = dists != float('inf')
        if np.sum(valid) < 10:
            print("!! Not enough overlapping ground points found.")
            z_diff = np.mean(pts_t[:, 2]) - np.mean(pts_s[:, 2])
            return 0.0, 0.0, z_diff

        x_s = pts_s[valid, 0]
        y_s = pts_s[valid, 1]
        z_s = pts_s[valid, 2]
        z_t = pts_t[indices[valid], 2]

        diff_z = z_t - z_s
        A = np.vstack([x_s, y_s, np.ones(len(x_s))]).T
        
        coeffs, _, _, _ = np.linalg.lstsq(A, diff_z, rcond=None)
        
        a, b, c = coeffs
        print(f":: Tilt Optimization Result -> Slope X(a): {a:.6f}, Slope Y(b): {b:.6f}, Shift Z(c): {c:.4f}")
        
        return a, b, c

    # -------------------------------------------------------------------------
    # ⑥ LAS保存
    # -------------------------------------------------------------------------
    def save_aligned_las(self, output_path, original_las, aligned_points):
        print(f"[Saving LAS] {output_path}")
        try:
            min_coords = np.min(aligned_points, axis=0)
            
            new_header = lp.LasHeader(point_format=original_las.header.point_format, version=original_las.header.version)
            new_header.offsets = min_coords
            new_header.scales = original_las.header.scales
            
            new_las = lp.LasData(header=new_header)
            new_las.x = aligned_points[:, 0]
            new_las.y = aligned_points[:, 1]
            new_las.z = aligned_points[:, 2]
            
            copy_dims = ['red', 'green', 'blue', 'intensity', 'classification']
            for dim in copy_dims:
                try:
                    data = getattr(original_las, dim)
                    setattr(new_las, dim, data)
                except:
                    pass
            
            new_las.write(output_path)
            print(":: Save successful.")
        except Exception as e:
            print(f"!! Error saving LAS: {e}")

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ORB-based LAS Registration with Fixed Resolution and ICP Refinement")
    parser.add_argument("source", help="Source LAS file")
    parser.add_argument("target", help="Target LAS file")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Downsample voxel size (m)")
    parser.add_argument("--pixel_size", type=float, default=0.045, help="Projection resolution in meters/pixel (default: 0.1m = 10cm)")
    args = parser.parse_args()
    

    myt_delta = datetime.timedelta(hours=8)
    MYT = datetime.timezone(myt_delta, 'MYT')
    timestamp = datetime.datetime.now(MYT).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"R_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f":: Output directory: {os.path.abspath(output_dir)}")


    source_name = os.path.splitext(os.path.basename(args.source))[0]
    target_name = os.path.splitext(os.path.basename(args.target))[0]
    base_filename = f"{source_name}_reg_to_{target_name}"
    final_las_path = os.path.join(output_dir, f"{base_filename}.las")
    
    # GlobalRegistrationの初期化時にピクセル解像度を渡す
    gr = GlobalRegistration(args.voxel_size, resolution=args.pixel_size)

    # 1. データの読み込み
    pcd_s, raw_las_s = gr.load_las_no_transform(args.source, keep_raw=False)
    pcd_t, _         = gr.load_las_no_transform(args.target, keep_raw=False)

    # 2. 地理的重複領域の抽出
    pcd_s_ov, pcd_t_ov = gr.extract_overlap_region(pcd_s, pcd_t, threshold=1.0)
    
    # 3. 地面除去 (CSF)
    print(":: Filtering ground with CSF...")
    s_ground, s_nonground = gr.filter_ground_csf(pcd_s_ov)
    t_ground, t_nonground = gr.filter_ground_csf(pcd_t_ov)

    # --- 最適なスライスの探索ループ ---
    best_score = float('inf')
    best_M = np.eye(3)
    best_height = 0.0
    
    height_slice_thickness = 0.2
    search_ranges = np.arange(0.7, 2.5, 0.2)

    print(f":: Searching best height slice ({len(search_ranges)} steps)...")

    for h in search_ranges:
        s_slice = gr.extract_slice(s_nonground, s_ground, h, h + height_slice_thickness)
        t_slice = gr.extract_slice(t_nonground, t_ground, h, h + height_slice_thickness)
        
        img_s, img_t = gr.pcd_to_image(s_slice, t_slice)
        
        # 4. ORB位置合わせ (Rigid with Fixed Scale)
        M_cand, inliers = gr.align_rigid_orb(img_s, img_t)
        
        if inliers > 10:
            score = gr.evaluate_mean(img_s, img_t, M_cand)
            print(f"   H={h:.1f}m: Error={score:.2f} px, Inliers={inliers}")
            
            if score < best_score:
                best_score = score
                best_M = M_cand
                best_height = h
        else:
            print(f"   H={h:.1f}m: Not enough matches.")

    print(f"\n:: Best Result -> Height: {best_height:.1f}m, Error: {best_score:.2f}")
    
    # 5. 最終的な適用と保存
    print(":: Applying transform to full resolution cloud...")
    
    raw_points = np.vstack((raw_las_s.x, raw_las_s.y, raw_las_s.z)).T
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(raw_points)
    
    # (A) XY平面の剛体変換適用 (ORB結果)
    print(":: Applying ORB Initial Alignment...")
    pcd_full_aligned = gr.apply_rigid_transform(pcd_full, best_M)
    
    # 地面点群（チルト計算用）と非地面点群（ICP用）にもORB結果を適用
    s_ground_aligned = gr.apply_rigid_transform(s_ground, best_M)
    s_nonground_aligned = gr.apply_rigid_transform(s_nonground, best_M)

    # (A-2) ICPによる微調整処理 (New Feature)
    # ORB結果で位置合わせされた点群から、再度ベストスライスを抽出
    s_slice_aligned = gr.extract_slice(s_nonground_aligned, s_ground_aligned, best_height, best_height + height_slice_thickness)
    t_slice_ref = gr.extract_slice(t_nonground, t_ground, best_height, best_height + height_slice_thickness)
    
    # ICPのしきい値は解像度の5倍程度を目安に設定 (例: 10cm解像度なら50cmまで許容して吸着)
    icp_threshold = args.pixel_size * 0.75
    M_icp_4x4 = gr.align_fine_icp(s_slice_aligned, t_slice_ref, threshold=icp_threshold)
    
    # ICPの結果を全体と地面点群に適用
    pcd_full_aligned.transform(M_icp_4x4)
    s_ground_aligned.transform(M_icp_4x4)

    # (B) Z軸の最適化 (Tilt & Shift)
    # ICP補正後の地面点群を使ってチルトを計算
    a, b, c = gr.calculate_tilt_and_shift(s_ground_aligned, t_ground)
    
    pts_final = np.asarray(pcd_full_aligned.points)
    z_adjustment = (a * pts_final[:, 0]) + (b * pts_final[:, 1]) + c
    pts_final[:, 2] += z_adjustment

    # 保存
    gr.save_aligned_las(final_las_path, raw_las_s, pts_final)

    # ---------------------------------------------------------
    # 画像による位置合わせ確認 (Overlay Image Output)
    # ---------------------------------------------------------
    print(":: Generating result overlay image...")
    
    pcd_final_vis = o3d.geometry.PointCloud()
    pcd_final_vis.points = o3d.utility.Vector3dVector(pts_final)

    # 確認用画像生成のため、最終結果から再度CSFとスライス
    # (すでに位置合わせ済みなのでそのまま抽出)
    s_ground_final, s_nonground_final = gr.filter_ground_csf(pcd_final_vis)
    s_slice_final = gr.extract_slice(s_nonground_final, s_ground_final, best_height, best_height + height_slice_thickness)
    
    # ターゲット側のスライスは元のものを使用
    t_slice_final = gr.extract_slice(t_nonground, t_ground, best_height, best_height + height_slice_thickness)

    # 既存の min_xy と 計算済み img_w/h を使って投影
    img_s_final, img_t_final = gr.pcd_to_image(s_slice_final, t_slice_final)

    h_img, w_img = img_t_final.shape
    overlay_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    overlay_img[..., 1] = img_s_final 
    overlay_img[..., 2] = img_t_final 

    img_output_path = os.path.join(output_dir, f"{base_filename}_overlay.png")
    cv2.imwrite(img_output_path, overlay_img)
    print(f"[Saved Image] {img_output_path}")
    

if __name__ == "__main__":
    main()