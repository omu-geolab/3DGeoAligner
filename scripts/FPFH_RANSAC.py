import laspy
import open3d as o3d
import numpy as np
import os

def load_las_as_o3d_pcd(las_path):
    """LASファイルを読み込んでOpen3DのPointCloudオブジェクトに変換する"""
    print(f"Loading: {os.path.basename(las_path)}")
    las = laspy.read(las_path)
    
    # 座標データの抽出 (X, Y, Z)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Open3Dの点群オブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 色情報(RGB)が含まれている場合は保持
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
    return pcd

def save_o3d_pcd_as_las(pcd, original_las_path, output_las_path):
    """Open3Dの点群データを、元のLASのヘッダー情報を引き継いでLASファイルとして保存する"""
    print(f"Saving registered source cloud to: {output_las_path}")
    
    # 変換後の座標を抽出
    points = np.asarray(pcd.points)
    
    # 元のLASファイルのヘッダー情報をコピーして新規ファイルを作成（データ構造を維持するため）
    original_las = laspy.read(original_las_path)
    new_header = original_las.header
    
    # 新しいLASオブジェクトの作成
    output_las = laspy.LasData(new_header)
    
    # 座標の書き込み
    output_las.x = points[:, 0]
    output_las.y = points[:, 1]
    output_las.z = points[:, 2]
    
    # 色情報が存在する場合は色も書き込む
    if pcd.has_colors():
        colors = np.asarray(pcd.colors) * 65535.0
        output_las.red = colors[:, 0].astype(np.uint16)
        output_las.green = colors[:, 1].astype(np.uint16)
        output_las.blue = colors[:, 2].astype(np.uint16)
        
    # ファイルへの書き込み実行
    output_las.write(output_las_path)
    print("Output completed successfully.")

def prepare_dataset(pcd, voxel_size):
    """ダウンサンプリング、法線推定、およびFPFH特徴量の計算を行う"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def execute_ransac_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """FPFHマッチングに基づくRANSAC初期位置合わせを実行する"""
    distance_threshold = voxel_size * 1.5
    print("-> Starting RANSAC global registration...")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(1.0),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 500)
    )
    return result

def main():
    # --- パラメータ設定 ---
    VOXEL_SIZE = 0.08
    
    # 入力ファイルパスの設定
    source_las_path = "data/raw/20251211_tondabayashi/epsg32653/scaniverse_tondabayashi_02.las"  # 動かしたい点群（未整合）
    target_las_path = "data/raw/20251211_tondabayashi/epsg32653/scaniverse_tondabayashi_01.las"   # 基準とする点群
    
    # 出力ファイルパスの指定 ★ここを任意のパスに変更してください
    output_las_path = "results/fpfh_ransac.las" 
    
    # 1. データ読み込み
    source = load_las_as_o3d_pcd(source_las_path)
    target = load_las_as_o3d_pcd(target_las_path)
    
    # 2. 前処理 (ダウンサンプリング & FPFH特徴量抽出)
    source_down, source_fpfh = prepare_dataset(source, VOXEL_SIZE)
    target_down, target_fpfh = prepare_dataset(target, VOXEL_SIZE)
    
    # 3. RANSACによるグローバル位置合わせ
    ransac_result = execute_ransac_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, VOXEL_SIZE
    )
    
    print("\n=== RANSAC Registration Result ===")
    print("Transformation Matrix:\n", ransac_result.transformation)
    
    # 4. 算出された変換行列(Rotation/Translation)を元の高密度なsource点群全体に適用
    print("-> Applying transformation matrix to the source point cloud...")
    source_transformed = source.transform(ransac_result.transformation)
    
    # 5. 整合後のsource点群をLASファイルとして出力
    save_o3d_pcd_as_las(source_transformed, source_las_path, output_las_path)

if __name__ == "__main__":
    main()