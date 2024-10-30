import glob
import open3d as o3d

読み込みたいファイルがあるディレクトリパスを指定
directory_path = "/Users/tsuchiyasatsukiha/SRC/データ/DJ06_PCD_Send_File_1"

# 指定したディレクトリからすべての .ply ファイルのパスを取得
file_paths = glob.glob(f"{directory_path}/*.ply")

# 各ファイルを順に開いて内容を表示
for file_path in file_paths:
    # ファイルを読み込む
    point_cloud = o3d.io.read_point_cloud(file_path)
    
    # 読み込んだ点群の情報を表示
    print(f"Loaded {file_path}")
    print(point_cloud)
    
    # 必要に応じて、点群を可視化
    o3d.visualization.draw_geometries([point_cloud])
