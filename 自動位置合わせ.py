import glob
import open3d as o3d
import numpy as np

# 読み込みたいファイルがあるディレクトリパスを指定
directory_path = "/Users/tsuchiyasatsukiha/SRC/データ/DJ06_PCD_Send_File_1"
file_paths = glob.glob(f"{directory_path}/*.ply")

# ファイルが少なくとも2つ以上あるかを確認
if len(file_paths) < 2:
    print("少なくとも2つの .ply ファイルが必要です。")
else:
    # 最初のファイルを基準点群として読み込む
    source_point_cloud = o3d.io.read_point_cloud(file_paths[0])
    print(f"Loaded base point cloud: {file_paths[0]}")

    # 2つ目以降のファイルに対して順に位置合わせを実施
    for i, file_path in enumerate(file_paths[1:], start=2):
        # 対象点群を読み込み
        target_point_cloud = o3d.io.read_point_cloud(file_path)
        print(f"Aligning {file_path} with base point cloud")

        # ICPアルゴリズムを使用して位置合わせを実行
        threshold = 0.02  # ICPの距離閾値（データに合わせて調整してください）
        trans_init=np.eye(4)

 # 点対点ICPによる位置合わせ
        reg_p2p = o3d.pipelines.registration.registration_icp(
            target_point_cloud, source_point_cloud, threshold,trans_init,

            o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # 点対点変換推定
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)  # 最大反復回数を設定
        )


        # 位置合わせ結果を適用
        target_point_cloud.transform(reg_p2p.transformation)
        print(f"Transformation matrix for {file_path}:\n{reg_p2p.transformation}")

        # 基準点群に対象点群をマージ
        source_point_cloud += target_point_cloud

    # 最終的なマージ結果を表示
    o3d.visualization.draw_geometries([source_point_cloud])
