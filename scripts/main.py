import argparse
import subprocess
import sys
import os
import json
import time

# プロジェクトのルートディレクトリ（GeoAligner/）の絶対パスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_script(script_name, args_list):
    """指定されたスクリプトをサブプロセスとして実行する"""
    script_path = os.path.join(BASE_DIR, 'scripts', script_name)
    
    # 実行するコマンドの構築 (python scripts/xxx.py arg1 arg2 ...)
    cmd = [sys.executable, script_path] + [str(a) for a in args_list]
    
    print("\n" + "="*60)
    print(f"▶ Executing: {' '.join(cmd)}")
    print("="*60)
    
    try:
        # cwd=BASE_DIR を指定することで、結果が常にGeoAligner/resultsに出力されるようにする
        subprocess.run(cmd, check=True, cwd=BASE_DIR)
    except subprocess.CalledProcessError as e:
        print(f"\n!! エラーが発生しました (Error Code: {e.returncode})")
        print(f"!! スクリプト {script_name} の実行に失敗しましたが、次の処理を継続します。")

def handle_batch(config_path):
    """JSON設定ファイルを読み込んで連続処理を実行する"""
    full_config_path = os.path.join(BASE_DIR, config_path)
    if not os.path.exists(full_config_path):
        print(f"!! バッチ設定ファイルが見つかりません: {full_config_path}")
        sys.exit(1)

    with open(full_config_path, 'r', encoding='utf-8') as f:
        jobs = json.load(f)

    print(f":: バッチ処理を開始します (合計 {len(jobs)} ジョブ)")
    
    for i, job in enumerate(jobs, 1):
        task_type = job.get('task')
        print(f"\n--- [Job {i}/{len(jobs)}] Task: {task_type} ---")
        
        if task_type == 'georeference':
            args = [
                job['las_path'],
                job['shp_path'],
                job['dem_path']
            ]
            if 'las_epsg' in job: args.extend(['--las_epsg', job['las_epsg']])
            if 'target_epsg' in job: args.extend(['--target_epsg', job['target_epsg']])
            if 'pixel_size' in job: args.extend(['--pixel_size', job['pixel_size']])
            if 'smooth_kernel' in job: args.extend(['--smooth_kernel', job['smooth_kernel']])
            
            run_script('georeference.py', args)

        elif task_type == 'registration':
            args = [
                job['source'],
                job['target']
            ]
            if 'voxel_size' in job: args.extend(['--voxel_size', job['voxel_size']])
            if 'pixel_size' in job: args.extend(['--pixel_size', job['pixel_size']])
            if 'out_dir' in job: args.extend(['--out_dir', job['out_dir']])
            
            run_script('registration.py', args)
        
        elif task_type == 'evaluation':
            args = [
                job['points_file'],
                job['matrix_file']
            ]
            # evaluation.py が受け取るオプション引数があれば追加します（以下は例です）
            if 'out_dir' in job: args.extend(['--out_dir', job['out_dir']])
            if 'direction' in job: args.extend(['--direction', job['direction']])            
            run_script('evaluation.py', args)
            
        else:
            print(f"!! 不明なタスクタイプです: {task_type}. スキップします。")

    print("\n:: 全てのバッチジョブが完了しました。")

def main():
    parser = argparse.ArgumentParser(description="GeoAligner: 幾何補正・位置合わせの統合管理基幹スクリプト")
    subparsers = parser.add_subparsers(dest='command', required=True, help='実行するモードを選択してください')

    # 単一実行モード: Georeference
    parser_geo = subparsers.add_parser('geo', help='georeference.py を単体実行します')
    parser_geo.add_argument('las_path')
    parser_geo.add_argument('shp_path')
    parser_geo.add_argument('dem_path')
    parser_geo.add_argument('--las_epsg', type=int, default=32653)
    parser_geo.add_argument('--target_epsg', type=int)
    parser_geo.add_argument('--pixel_size', type=float, default=0.1)

    # 単一実行モード: Registration
    parser_reg = subparsers.add_parser('reg', help='registration.py を単体実行します')
    parser_reg.add_argument('source')
    parser_reg.add_argument('target')
    parser_reg.add_argument('--voxel_size', type=float, default=0.005)
    parser_reg.add_argument('--pixel_size', type=float, default=0.045)
    parser_reg.add_argument('--out_dir', type=str, help='指定した共通フォルダ内の R_日時 フォルダに保存します')

    parser_eval = subparsers.add_parser('eval', help='evaluation.py を単体実行します')
    parser_eval.add_argument('points_file')
    parser_eval.add_argument('matrix_file', help='変換行列のTXTファイル')
    parser_eval.add_argument('--out_dir', type=str, default='results/', help='結果レポートの出力先ディレクトリ')
    parser_eval.add_argument('--direction', choices=['auto', 'left_to_right', 'right_to_left'], default='auto')
    # バッチ実行モード
    parser_batch = subparsers.add_parser('batch', help='JSONファイルから複数のタスクを連続実行します')
    parser_batch.add_argument('config_path', help='バッチ処理の設定ファイル(JSON)へのパス (例: data/raw/batch_jobs.json)')

    args, unknown = parser.parse_known_args()

    start_time = time.time()

    # コマンドごとの分岐処理
    if args.command == 'geo':
        run_script('georeference.py', sys.argv[2:])
    elif args.command == 'reg':
        run_script('registration.py', sys.argv[2:])
    elif args.command == 'eval':
        run_script('evaluation.py', sys.argv[2:])
    elif args.command == 'batch':
        handle_batch(args.config_path)

    end_time = time.time()
    print(f"\n[Total process Time: {end_time - start_time:.2f}] s")

if __name__ == "__main__":
    main()