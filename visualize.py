import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Caminho para o ficheiro .xyzc de entrada")
args, _ = parser.parse_known_args()

file_path = args.input
data = np.loadtxt(file_path)
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


pasta_destino = os.path.dirname(args.input)  # ex: "assets"
nome_base = os.path.basename(args.input)     # ex: "mini_verticalPlanes_downs_segmented.xyzc"

# 2. Remover a extensão (.xyzc)
nome_sem_extensao = os.path.splitext(nome_base)[0]

# 3. Remover os sufixos (downs_segmented) e juntar o resto com '_'
# O join junta a lista de volta numa string perfeitamente formatada
partes_nome = nome_sem_extensao.split('_')[:-2]
nome_limpo = "_".join(partes_nome)

# 4. Construir o caminho final seguro
caminho_output = os.path.join(pasta_destino, f"{nome_limpo}.html")

fig.write_html(caminho_output)
print(f"Visualização guardada com sucesso em: {caminho_output}")
