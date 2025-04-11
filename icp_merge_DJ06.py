import glob
import open3d as o3d
import numpy as np
import config
import copy
import re
import os

import re
import os

def extract_angles(file_path):
    """
    file_pathから角度を抽出する関数
    arg:
        file_path : ファイルパス
    """
    match = re.search(r"_(\d+)-(\d+)\.ply", os.path.basename(file_path))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def icp_merge(directory_path, output_directory):
    """
    隣り合う点群データを貼り合わせて3D復元し、保存する関数

    Args:
        directory_path (str): 入力点群データのディレクトリパス
        output_directory (str): 結果を保存するディレクトリパス
    """

    # 出力ディレクトリを作成（存在しない場合）
    os.makedirs(output_directory, exist_ok=True)

    # 読み込みたい PLY ファイルのリストを取得
    file_paths = glob.glob(f"{directory_path}/*.ply")

    # 角度順にソート
    file_paths.sort(key=lambda x: extract_angles(x))

    # ファイルが少なくとも2つ以上あるかを確認
    if len(file_paths) < 2:
        print("少なくとも2つの .ply ファイルが必要です。")
        return

    # パラメータ設定
    voxel_size = 0.3  # ダウンサンプリングのボクセルサイズ
    threshold = 0.5   # ICP の距離閾値

    # ICPカスタマイズ
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=2000,
        relative_fitness=1e-10,
        relative_rmse=1e-10
    )

    # 累積変換行列
    transformations = [np.eye(4)]  # 初期の変換行列は単位行列

    # 隣接する点群に対して ICP を適用
    for i in range(len(file_paths) - 1):
        print(f"Registering file {i+1} to file {i+2}...")

        # 点群の読み込みとダウンサンプリング
        target_pc = o3d.io.read_point_cloud(file_paths[i]).voxel_down_sample(voxel_size)
        source_pc = o3d.io.read_point_cloud(file_paths[i + 1]).voxel_down_sample(voxel_size)

        # ICP で位置合わせ
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pc, target_pc, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria
        )

        # 累積変換行列を更新
        transformations.append(transformations[-1] @ reg_p2p.transformation)

        print("Done.")

    # 全体の位置合わせ結果を適用し、点群をマージ
    final_pc = o3d.geometry.PointCloud()

    print("Merging point clouds...")

    for i, file_path in enumerate(file_paths):
        pc = o3d.io.read_point_cloud(file_path).voxel_down_sample(0.7)
        pc.transform(transformations[i])
        final_pc += pc

    print("Final merge complete.")

    # 結果を表示
    o3d.visualization.draw_geometries([final_pc])

    # **結果を保存**
    output_file = os.path.join(output_directory, "merged_point_cloud.ply")
    o3d.io.write_point_cloud(output_file, final_pc)
    print(f"Saved merged point cloud to: {output_file}")

# **ディレクトリパス設定**
input_directory = config.input_directory_path
output_directory = "/Users/tsuchiyasatsukiha/SRC/データ/隣接位置合わせ/"

# **関数の実行**
icp_merge(input_directory, output_directory)
