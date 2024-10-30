import open3d as o3d
import copy
import numpy as np
import config

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0, 1])  # ICP後のソース点群は青
    target_temp.paint_uniform_color([0, 1, 0])  # ICP後のソース点群は青
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    
def draw_registration_result_smaller_pixel(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0, 1])  # ソース点群を青に設定
    target_temp.paint_uniform_color([0, 1, 0])  # ターゲット点群を緑に設定
    source_temp.transform(transformation)  # 変換後のソース点群

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)

    # 点のサイズを変更する
    opt = vis.get_render_option()
    opt.point_size = 1.0  # ここで点のサイズを調整 (デフォルトは 5.0)

    vis.run()
    vis.destroy_window()


# 初期準備
# demo_icp_pcds = o3d.data.DemoICPPointClouds()

# ## demo用のデータ
# source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
# target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

# 入力データ
source = o3d.io.read_point_cloud(config.input_path2)
target = o3d.io.read_point_cloud(config.input_path)

# まずは点群データを正しく読み込んでいるかを確認
if not source.has_points():
    print("source point cloud has no points.")
if not target.has_points():
    print("target point cloud has no points.")

# ターゲットの法線ベクトルを計算
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 必要に応じて、ソースの点群にも法線を計算
source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# ポイント・トゥ・ポイントICP
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance=0.05,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# ポイント・トゥ・プレーンICP
reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance=0.05,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print("Transformation is:", reg_p2l.transformation)

# icpの対応点としての許容誤差
threshold = 0.1

# # 初期変換
# trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
#                          [-0.139, 0.967, -0.215, 0.7],
#                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
trans_init = np.eye(4)

draw_registration_result_smaller_pixel(source, target, trans_init)

# icpカスタマイズ
criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    max_iteration=2000,      # 最大反復回数
    relative_fitness=1e-6,   # 相対フィットネスの閾値
    # relative_fitness=1e-6,
    relative_rmse=1e-6       # 相対RMSEの閾値
    # relative_rmse=1e-6
)

# point-point ICP
print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria)
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result_smaller_pixel(source, target, reg_p2p.transformation)

## 点対面のicp
# print("Apply point-to-plane ICP")
# reg_p2l = o3d.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init,
#     o3d.pipelines.registration.TransformationEstimationPointToPlane())
# print(reg_p2l)
# print("Transformation is:")
# print(reg_p2l.transformation)
# draw_registration_result_smaller_pixel(source, target, reg_p2l.transformation)