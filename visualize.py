import argparse
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Caminho para o ficheiro .xyzc de entrada")
args, _ = parser.parse_known_args()

data = np.loadtxt(args.input)
X, Y, Z, Cluster = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=("Objeto Inicial (Original)", "Objeto Segmentado pela IA")
)

fig.add_trace(
    go.Scatter3d(x=X, y=Y, z=Z, mode='markers',
                 marker=dict(size=2.5, color='royalblue', opacity=0.8), name="Original"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter3d(x=X, y=Y, z=Z, mode='markers',
                 marker=dict(size=2.5, color=Cluster, colorscale='Turbo', opacity=0.8), name="Segmentado"),
    row=1, col=2
)

fig.update_layout(title_text="Comparativo ParseNet: Antes vs Depois", height=600, showlegend=False)

# Guardar imagem JPG
base_name = os.path.splitext(os.path.basename(args.input))[0]
image_name = base_name.replace("_down_segmented", "") + ".jpg"
os.makedirs("imagens", exist_ok=True)
output_image_path = os.path.join("imagens", image_name)

try:
    fig.write_image(output_image_path)
    print(f"Imagem guardada em: {output_image_path}")
except Exception as e:
    print(f"Aviso: não foi possível guardar a imagem. Instala kaleido: pip install kaleido")

fig.show()
