import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Caminho para o ficheiro .xyzc de entrada")
args, _ = parser.parse_known_args()

# 1. Carregar os dados
data = np.loadtxt(args.input)

X, Y, Z, Cluster = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# 2. Criar uma estrutura de subplots 3D (1 linha, 2 colunas)
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=("Objeto Inicial (Original)", "Objeto Segmentado pela IA")
)

# 3. Adicionar o objeto inicial (Gráfico da Esquerda - Cor Única)
fig.add_trace(
    go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(size=2.5, color='royalblue', opacity=0.8),
        name="Original"
    ),
    row=1, col=1
)

# 4. Adicionar o objeto segmentado (Gráfico da Direita - Colorido por IDs)
fig.add_trace(
    go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=2.5,
            color=Cluster,
            colorscale='Turbo', # Paleta de cores bem distintas para os segmentos
            opacity=0.8
        ),
        name="Segmentado"
    ),
    row=1, col=2
)

# 5. Ajustar o layout para exibição
fig.update_layout(
    title_text="Comparativo ParseNet: Antes vs Depois",
    height=600,
    showlegend=False
)

fig.show()