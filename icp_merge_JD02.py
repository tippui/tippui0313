import glob
import open3d as o3d
import numpy as np
import config
import copy
import re
import os

"""内容
    # JD02_PCDの隣り合う点群に対してicpを行い、3D復元を行うスクリプト
"""

def draw_registration_result_smaller_pixel(source, target, window_name):
    """
    point_cloudの描画関数
    arg:
        source : sourceのpoint_cloud
        target : targetのpoint_cloud
        window_name : 描画ウインドウの名前を指定できる

    """

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0, 1])  # ソース点群を青に設定
    target_temp.paint_uniform_color([1, 0, 0])  # ターゲット点群を緑に設定

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name = window_name)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)

    # 点のサイズを変更する
    opt = vis.get_render_option()
    opt.point_size = 5.0  # ここで点のサイズを調整 (デフォルトは 5.0)

    vis.run()
    vis.destroy_window()

def extract_angles_no_leading_zeros(file_path):
    """
    file_pathから角度を抽出する関数
    arg:
        file_path : ファイルパス
    """

    match = re.search(r"_(\d+)-(\d+)", os.path.basename(file_path))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def icp_merge(directory_path):
    """
    directory_pathの中の隣り合う点群データを貼り合わせて3D復元する関数
    arg:
        directory_path : ディレクトリパス
    """

    # 初期準備
    demo_icp_pcds = o3d.data.DemoICPPointClouds()


    # 読み込みたいファイルがあるディレクトリパスを指定
    directory_path = directory_path
    file_paths = glob.glob(f"{directory_path}/*.ply")

    # まず最初の角度、次に次の角度でソートする
    file_paths.sort()

    # 隣り合う物体順にソート
    # Start sorting based on minimal angle differences
    sorted_file_paths = [file_paths[0]]  # Begin with the first file
    remaining_files = file_paths[1:]

    while remaining_files:
        last_angle = extract_angles_no_leading_zeros(sorted_file_paths[-1])
        # Find the next file with the smallest angle difference
        next_file = min(remaining_files, key=lambda x: sum(abs(a - b) for a, b in zip(last_angle, extract_angles_no_leading_zeros(x))))
        sorted_file_paths.append(next_file)
        remaining_files.remove(next_file)


    # ファイルが少なくとも2つ以上あるかを確認
    if len(sorted_file_paths) < 2:
        print("少なくとも2つの .ply ファイルが必要です。")
    else:
        # ダウンサンプリングのボクセルサイズ
        voxel_size = 0.2
        
        # ICPの距離閾値
        threshold = 0.5

        # 点群を順に読み込んで、位置合わせを行う
        transformed_pcs = []
        transformations = [np.eye(4)]  # 初期の変換行列は単位行列

        # icpカスタマイズ
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        # 最大反復回数
        max_iteration=2000,      
        # 相対フィットネスの閾値
        relative_fitness=1e-10,   
        # 相対RMSEの閾値
        relative_rmse=1e-10       
    )
        



        for i in range(len(sorted_file_paths) - 1):

            # デバッグ用
            print(f"ragistrate_file_number{i+1}_to_file_number{i+2}")

            # i 番目と i+1 番目の点群を読み込む
            target_pc = o3d.io.read_point_cloud(sorted_file_paths[i]).voxel_down_sample(voxel_size)
            source_pc = o3d.io.read_point_cloud(sorted_file_paths[i + 1]).voxel_down_sample(voxel_size)

            # 位置合わせ前を表示
            # draw_registration_result_smaller_pixel(source_pc,target_pc, window_name="before...")


            # i番目の点群をi+1番目の点群に、ICP位置合わせを行う
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pc, target_pc, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria
            )

            # 累積変換行列に対して、隣接位置合わせの結果を適用
            transformations.append(transformations[-1] @ reg_p2p.transformation)

            # i番目の点群をi+1番目の点群に、ICP位置合わせを行った結果を表示
            # if len(transformations) > 1 : 
            #     source_pc.transform(transformations[-1])
            # if len(transformations) > 2 :
            #     target_pc.transform(transformations[-2])
            # draw_registration_result_smaller_pixel(source_pc,target_pc, window_name="after...")

            # デバッグ用
            print(f"Done...")
            print("")


        # 全体の位置合わせ結果を適用し、点群をマージ
        final_pc = o3d.geometry.PointCloud()

        print("merge...")

        # 回転行列を元の点群に適用する
        for i, file_path in enumerate(sorted_file_paths):
            # pc = o3d.io.read_point_cloud(file_path)
            voxel_size = 0.7
            pc = o3d.io.read_point_cloud(file_path).voxel_down_sample(voxel_size)
            pc.transform(transformations[i])
            final_pc += pc

        print(f"Done...")

        # マージ結果を表示
        o3d.visualization.draw_geometries([final_pc])

        # TODO: 先生にこの方法で間違いがないかどうか聞いてみる！！、
        # TODO: いけた。。。


# ディレクトリパスの指定
directory_path = config.input_directory_path2

# 関数の実行
icp_merge(directory_path=directory_path)

# 表示後にPLYファイルとして保存
o3d.io.write_point_cloud("merged_point_cloud.ply", final_pc)
print("点群データをmerged_point_cloud.plyとして保存しました")

