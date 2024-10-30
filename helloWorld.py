import open3d as o3d
import numpy as np

# Plyファイルから点群を読み込み
source_file = "C:\\Users\\thais\\Downloads\\DJ06_PCD_Send_File_1\\DJ06_PCD_Send_File_1\\241011_JD06_90-55.ply"
target_file = "C:\\Users\\thais\\Downloads\\DJ06_PCD_Send_File_1\\DJ06_PCD_Send_File_1\\241011_JD06_90-15.ply"

source_pcd = o3d.io.read_point_cloud(source_file)
target_pcd = o3d.io.read_point_cloud(target_file)

# 点群を可視化（オプション）
o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="Before Alignment")

# ICPアルゴリズムで位置合わせ
threshold = 0.02  # 距離しきい値
trans_init = np.eye(4)  # 初期位置合わせ行列
icp_result = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# 位置合わせ後の点群
source_pcd.transform(icp_result.transformation)   

# 位置合わせ結果の表示
o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="After Alignment")

# ICP結果の出力
print("Transformation matrix:")
print(icp_result.transformation)
