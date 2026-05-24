import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Caminho para o ficheiro .xyzc de entrada")
args, _ = parser.parse_known_args()

data = np.loadtxt(args.input)
X, Y, Z, Cluster = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
points = np.column_stack([X, Y, Z])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="ParseNet Segmentation", width=1400, height=600)

# Nuvem original (azul)
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(points)
pcd_original.paint_uniform_color([0.1, 0.4, 0.9])
vis.add_geometry(pcd_original)

# Nuvem segmentada (cores por cluster)
pcd_segmented = o3d.geometry.PointCloud()
pcd_segmented.points = o3d.utility.Vector3dVector(points)
colors = plt.cm.turbo(Cluster / Cluster.max())[:, :3]
pcd_segmented.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(pcd_segmented)

# Guardar imagem
base_name = os.path.splitext(os.path.basename(args.input))[0]
image_name = base_name.replace("_down_segmented", "") + ".jpg"
os.makedirs("imagens", exist_ok=True)
output_image_path = os.path.join("imagens", image_name)

vis.capture_screen_image(output_image_path)
print(f"Imagem guardada em: {output_image_path}")

vis.run()
vis.destroy_window()
