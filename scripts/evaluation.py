import argparse
import numpy as np
import os
import re
import sys

def load_correspondences(filepath):
    """
    対応点テキストファイルを読み込み、左側の点群と右側の点群に分ける。
    フォーマット想定: ID_A X_A Y_A Z_A ID_B X_B Y_B Z_B
    """
    left_pts, right_pts = [], []
    left_ids, right_ids = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                try:
                    # ヘッダー行などの文字列をスキップし、数値データのみを抽出
                    x1, y1, z1 = float(parts[1]), float(parts[2]), float(parts[3])
                    x2, y2, z2 = float(parts[5]), float(parts[6]), float(parts[7])
                    left_ids.append(parts[0])
                    left_pts.append([x1, y1, z1])
                    right_ids.append(parts[4])
                    right_pts.append([x2, y2, z2])
                except ValueError:
                    continue
    return np.array(left_ids), np.array(left_pts), np.array(right_ids), np.array(right_pts)

def load_transformation_matrix(filepath):
    """4x4の変換行列を読み込む"""
    try:
        matrix = np.loadtxt(filepath, delimiter=',')
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix shape is not 4x4: {matrix.shape}")
        return matrix
    except Exception as e:
        print(f"Error loading matrix: {e}")
        sys.exit(1)

def apply_transformation(pts, matrix):
    """3D点群に4x4変換行列を適用する"""
    pts_homo = np.hstack((pts, np.ones((pts.shape[0], 1))))
    transformed = (matrix @ pts_homo.T).T
    return transformed[:, :3]

def calc_errors(pts1, pts2):
    """点群間の距離(3D, 2D(XY), Z)を計算する"""
    diff = pts1 - pts2
    dist_3d = np.linalg.norm(diff, axis=1)
    dist_2d = np.linalg.norm(diff[:, :2], axis=1)
    dist_z = np.abs(diff[:, 2])
    return dist_3d, dist_2d, dist_z

def get_stats(dists):
    """距離の配列から各種統計量(論文用)を算出する"""
    if len(dists) == 0:
        return {'rmse': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    return {
        'rmse': np.sqrt(np.mean(dists**2)),
        'mean': np.mean(dists),
        'std': np.std(dists),
        'min': np.min(dists),
        'max': np.max(dists)
    }

def main():
    parser = argparse.ArgumentParser(description="対応点と変換行列を用いてレジストレーション精度を評価します。")
    parser.add_argument("points_file", help="対応点データのTXTファイル (例: 01-02.txt)")
    parser.add_argument("matrix_file", help="変換行列のTXTファイル (例: 2026...01_reg2_2026...02_matrix.txt)")
    parser.add_argument("--out_dir", type=str, default="results/", help="結果レポートの出力先ディレクトリ")
    parser.add_argument("--direction", choices=["auto", "left_to_right", "right_to_left"], default="auto",
                        help="行列を適用する方向。autoの場合はファイル名から自動推測します。")
    
    args = parser.parse_args()

    # 1. データの読み込み
    left_ids, left_pts, right_ids, right_pts = load_correspondences(args.points_file)
    if len(left_pts) == 0:
        print(f"!! Error: {args.points_file} から有効な対応点を読み込めませんでした。")
        sys.exit(1)
        
    matrix = load_transformation_matrix(args.matrix_file)

    # 2. SourceとTargetの対応方向の決定
    direction = args.direction
    if direction == "auto":
        # points_fileから番号(例: '01', '02')を抽出
        m = re.search(r'(\d+)-(\d+)', os.path.basename(args.points_file))
        if m:
            left_id, right_id = m.groups()
            matrix_base = os.path.basename(args.matrix_file)
            if "_reg2_" in matrix_base:
                src_part, tgt_part = matrix_base.split("_reg2_")
                
                # 日付（20260125など）の一部への誤反応を防ぐための判定関数
                # 「_01」のように直前にアンダースコアがあり、直後が _ または . または文末 の場合のみマッチ
                def has_id(text, target_id):
                    return re.search(rf'_{target_id}(?:_|\.|$)', text) is not None

                # 行列のSource/Targetと対応点の左右が一致するか確認
                if has_id(src_part, left_id) and has_id(tgt_part, right_id):
                    direction = "left_to_right"
                elif has_id(src_part, right_id) and has_id(tgt_part, left_id):
                    direction = "right_to_left"

        if direction == "auto":
            print(":: 警告: ファイル名からSource->Targetの方向を推測できませんでした。デフォルト(左側の点をSource)で処理します。")
            direction = "left_to_right"
            
    print(f":: Direction mode: {direction}")

    # 方向に応じて適用するポイントを振り分け
    if direction == "left_to_right":
        src_ids, src_pts = left_ids, left_pts
        tgt_ids, tgt_pts = right_ids, right_pts
        src_label, tgt_label = "Left", "Right"
    else:
        src_ids, src_pts = right_ids, right_pts
        tgt_ids, tgt_pts = left_ids, left_pts
        src_label, tgt_label = "Right", "Left"

    # 3. 変換の適用
    trans_src_pts = apply_transformation(src_pts, matrix)

    # 4. 誤差計算
    dist_3d, dist_2d, dist_z = calc_errors(trans_src_pts, tgt_pts)

    # 5. レポートファイルへの出力
    os.makedirs(args.out_dir, exist_ok=True)
    
    points_base = os.path.splitext(os.path.basename(args.points_file))[0]
    matrix_base = os.path.splitext(os.path.basename(args.matrix_file))[0]
    out_filename = f"eval_report_{points_base}_{matrix_base}.txt"
    out_filepath = os.path.join(args.out_dir, out_filename)

    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write(" Registration Evaluation Report\n")
        f.write("============================================================\n")
        f.write(f"Points File    : {os.path.basename(args.points_file)}\n")
        f.write(f"Matrix File    : {os.path.basename(args.matrix_file)}\n")
        f.write(f"Apply Direction: {src_label} column -> {tgt_label} column\n")
        f.write(f"Total Points   : {len(src_pts)}\n\n")

        f.write("[Transformation Matrix]\n")
        for row in matrix:
            f.write(", ".join([f"{v:.8f}" for v in row]) + "\n")
        f.write("\n")

        f.write("[Statistical Results]\n")
        metrics = {"3D Error": dist_3d, "2D Error (XY)": dist_2d, "Z Error": dist_z}
        for name, dists in metrics.items():
            st = get_stats(dists)
            f.write(f"{name}:\n")
            f.write(f"  RMSE: {st['rmse']:.6f} m\n")
            f.write(f"  Mean: {st['mean']:.6f} m\n")
            f.write(f"  Std : {st['std']:.6f} m\n")
            f.write(f"  Min : {st['min']:.6f} m\n")
            f.write(f"  Max : {st['max']:.6f} m\n\n")

        f.write("[Point-to-Point Details]\n")
        # タブ区切りで表形式にして出力（Excel等へのコピーが容易です）
        f.write("Src_ID\tTgt_ID\tTrans_Src_X\tTrans_Src_Y\tTrans_Src_Z\tTgt_X\tTgt_Y\tTgt_Z\tDist_3D\tDist_2D\tDist_Z\n")
        for i in range(len(src_pts)):
            ts = trans_src_pts[i]
            tt = tgt_pts[i]
            d3, d2, dz = dist_3d[i], dist_2d[i], dist_z[i]
            f.write(f"{src_ids[i]}\t{tgt_ids[i]}\t{ts[0]:.4f}\t{ts[1]:.4f}\t{ts[2]:.4f}\t"
                    f"{tt[0]:.4f}\t{tt[1]:.4f}\t{tt[2]:.4f}\t{d3:.4f}\t{d2:.4f}\t{dz:.4f}\n")

    print(f":: Evaluation complete. Report saved to: {out_filepath}")

if __name__ == "__main__":
    main()