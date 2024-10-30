import open3d as o3d
import numpy as np
import config 

def save_visualization_from_multiple_angles(pcd_list, filepath, angles):
    """
    pointcloudを任意のカメラ位置から画像保存する関数
    param
    pcd_list: PointCloudのリスト
    filename: 任意のfilename
    angles: カメラ角度のリスト[(0,0), (30,0), (60,0) ...]のように指定する。
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # ウィンドウを非表示にする
    for pcd in pcd_list:
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)  # 各点群を更新
    
    vis.poll_events()
    vis.update_renderer()
    
    # カメラ設定を取得
    view_control = vis.get_view_control()
    
    for i, angle in enumerate(angles):
        # カメラを回転させる (回転軸や角度は必要に応じて調整可能)
        view_control.rotate(angle[0], angle[1])  # X方向とY方向の回転を指定
        vis.poll_events()
        vis.update_renderer()
        
        # 各角度ごとに画像を保存
        filename = filepath.format(i)
        vis.capture_screen_image(filename, do_render=True)
    
    vis.destroy_window()  # ウィンドウを閉じる

import open3d as o3d
import numpy as np

def icp(source, target, max_iterations=20, tolerance=1e-8):
    """
    簡易的なICPアルゴリズムの実装
    :param source: 元の点群 (open3d.geometry.PointCloud)
    :param target: ターゲットの点群 (open3d.geometry.PointCloud)
    :param max_iterations: 最大反復回数
    :param tolerance: 収束の閾値
    :return: 最終的な変換行列と収束した点群
    """
    # 初期変換行列の定義（初期位置をできるだけ合わせる必要がある）
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], 
                             [0.0, 0.0, 0.0, 1.0]])

    # ソース点群に初期変換を適用するためのコピーを作成
    source_copy = source
    source_copy = source_copy.transform(trans_init)

    
    # 剛体変換後の状態を確認
    # 点群の可視化（icp後）
    o3d.visualization.draw_geometries([source_copy, target_pcd],
                                    window_name="Point Cloud",
                                    width=800, height=600)

    # 変換行列を trans_init に設定
    transformation_matrix = trans_init

    # KDTreeを使ってターゲット点群から最近傍点を検索
    target_tree = o3d.geometry.KDTreeFlann(target)

    prev_error = float('inf')

    for i in range(max_iterations):

        # iteration count
        print(f"iteration {i + 1}.")

        # 各点の対応する最近傍点を見つける
        correspondences = []
        for point in source_copy.points:
            _, idx, _ = target_tree.search_knn_vector_3d(point, 1)
            correspondences.append(target.points[idx[0]])

        # 対応点のペアを配列に変換
        correspondences = np.array(correspondences)
        source_points = np.asarray(source_copy.points)

        # 変換行列を計算（最適な回転と平行移動）
        centroids_source = np.mean(source_points, axis=0)
        centroids_target = np.mean(correspondences, axis=0)

        source_centered = source_points - centroids_source
        target_centered = correspondences - centroids_target

        H = np.dot(source_centered.T, target_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # 回転行列が反転している場合、反転を防ぐ
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroids_target - np.dot(R, centroids_source)

        # 新しい変換行列を作成
        current_transformation = np.eye(4)
        current_transformation[:3, :3] = R
        current_transformation[:3, 3] = t

        # 変換行列を更新
        transformation_matrix = np.dot(current_transformation, transformation_matrix)

        # 点群に変換を適用
        source_copy.transform(current_transformation)

        # 誤差を計算
        error = np.mean(np.linalg.norm(np.asarray(source_copy.points) - correspondences, axis=1))

        print(f"error{i:1} : {error}")

        if abs(prev_error - error) < tolerance:
            print(f'ICP converged at iteration {i + 1}')
            break

        prev_error = error

        # visualize_every_iteration
        o3d.visualization.draw_geometries([source_copy, target_pcd],
                                    window_name=f"icp_iteration{i+1}",
                                    width=800, height=600)



    return transformation_matrix, source_copy

source_pcd = o3d.data.DemoICPPointClouds().paths[0]  # サンプルデータのソース点群
target_pcd = o3d.data.DemoICPPointClouds().paths[1]  # サンプルデータのターゲット点群

# 点群の読み込み（サンプルデータ）
source_pcd = o3d.io.read_point_cloud(source_pcd)  # 元の点群
target_pcd = o3d.io.read_point_cloud(target_pcd)  # ターゲット点群

# ソース点群をオレンジ、ターゲット点群を青に色付け
source_pcd.paint_uniform_color([1, 0.706, 0])  # オレンジ色
target_pcd.paint_uniform_color([0, 0.651, 0.929])  # 青色

# 座標軸を作成 (原点に表示)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# 点群の可視化（icp前）source
o3d.visualization.draw_geometries([source_pcd,coordinate_frame],
                                  window_name="Point Cloud",
                                  width=800, height=600)

# 点群の可視化（icp前）target
o3d.visualization.draw_geometries([target_pcd,coordinate_frame],
                                  window_name="Point Cloud",
                                  width=800, height=600)

# 点群の可視化（icp前）
o3d.visualization.draw_geometries([source_pcd, target_pcd,coordinate_frame],
                                  window_name="Point Cloud",
                                  width=800, height=600)

# ICPを実行
transformation, aligned_source = icp(source_pcd, target_pcd)

# 結果を表示
print("Transformation Matrix:")
print(transformation)

aligned_source.paint_uniform_color([0, 0, 1])  # ICP後のソース点群は青

# 点群の可視化（icp後）
o3d.visualization.draw_geometries([aligned_source, target_pcd,coordinate_frame],
                                  window_name="Point Cloud",
                                  width=800, height=600)

# さまざまな角度からの画像を保存
angles = [(0, 0), (90, 0), (180, 0), (270, 0), (360, 0), (450, 0), (540, 0)]  # X方向とY方向の回転角度
save_visualization_from_multiple_angles([source_pcd, target_pcd], config.output_path + "\icp_before_{}.png", angles)
