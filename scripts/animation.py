#このファイルでデモプレイのアニメーションを動かす。
#作成中

import argparse
import numpy as np
import laspy
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. データの読み込みと軽量化
# ==========================================
def load_and_sample_las(file_path, max_points=5000):
    print(f"Reading {file_path} ...")
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # 描画を軽くするためランダムに間引く
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    return points

def load_matrix(file_path):
    # カンマ区切りのテキストファイルに対応
    try:
        return np.loadtxt(file_path, delimiter=',')
    except ValueError:
        return np.loadtxt(file_path)

# ==========================================
# 2. 【最重要】「じわじわ寄せる」ための行列補間
# ==========================================
def generate_interpolated_transforms(T_final, num_steps=50):
    """初期位置から最終位置まで滑らかに推移する変換行列のリストを作成"""
    r = R.from_matrix(T_final[:3, :3])
    rotvec = r.as_rotvec()
    trans = T_final[:3, 3]

    transforms = []
    # 0% から 100% まで少しずつ変換行列を作る
    for i in range(num_steps + 1):
        fraction = i / num_steps
        T_step = np.eye(4)
        T_step[:3, :3] = R.from_rotvec(rotvec * fraction).as_matrix()
        T_step[:3, 3] = trans * fraction
        transforms.append(T_step)
        
    return transforms

def apply_transform(points, T):
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))
    transformed = (T @ points_h.T).T
    return transformed[:, :3]

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("-s", "--source", required=True)
    parser.add_argument("-m", "--matrix", required=True)
    parser.add_argument("-o", "--output", default="./results/sample_20260513/animation.gif")
    parser.add_argument("--steps", type=int, default=50, help="フレーム数（多いほどゆっくり動く）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    target_pts = load_and_sample_las(args.target, max_points=5000)
    source_pts = load_and_sample_las(args.source, max_points=5000)
    T_final = load_matrix(args.matrix)

    # 巨大座標対策
    center = np.mean(target_pts, axis=0)
    target_pts -= center
    source_pts -= center

    # じわじわ動かすための行列リストを取得
    transforms = generate_interpolated_transforms(T_final, num_steps=args.steps)

    print("描画の準備をしています...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 【視点が遠すぎる問題の解決】
    # 外れ値を無視して、点群が密集している部分にズームインする
    p_min = np.percentile(target_pts, 5, axis=0)
    p_max = np.percentile(target_pts, 95, axis=0)
    mid = (p_max + p_min) / 2.0
    max_range = np.max(p_max - p_min) / 2.0 * 1.2 
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_aspect('equal')
    ax.axis('off') 

    # Target（固定・青色）
    ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], c='#00A6ED', s=1, alpha=0.5)
    # Source（動く・オレンジ色）
    scatter_source = ax.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], c='#F6A800', s=1, alpha=0.8)

    # 1フレームごとの更新処理
    def update(frame_idx):
        if frame_idx % 10 == 0:
            print(f"フレーム生成中: {frame_idx}/{args.steps}")
        
        # 途中の変換行列を適用して表示を更新
        T = transforms[frame_idx]
        current_source_pts = apply_transform(source_pts, T)
        scatter_source._offsets3d = (current_source_pts[:, 0], current_source_pts[:, 1], current_source_pts[:, 2])
        return scatter_source,

    print(f"GIFアニメーションを書き出しています: {args.output}")
    # interval=80 (ミリ秒) で滑らかさを調整
    ani = animation.FuncAnimation(fig, update, frames=len(transforms), interval=80, blit=False)
    ani.save(args.output, writer='pillow')
    print("完了しました！")

if __name__ == "__main__":
    main()