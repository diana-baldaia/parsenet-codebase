import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Caminho do ficheiro .xyz de entrada")
args, _ = parser.parse_known_args()  # parse_known_args para compatibilidade com Colab

caminho_original = args.input
caminho_output = os.path.splitext(args.input)[0] + "_downs.xyz"


if os.path.exists(caminho_original):
    print(f"A abrir o ficheiro gigante {args.input}...")

    pontos = []
    with open(caminho_original, 'r') as f:
        for i, linha in enumerate(f):
            partes = linha.strip().split()
            if len(partes) >= 3:
                pontos.append([float(partes[0]), float(partes[1]), float(partes[2])])
            if i > 5000000:
                break

    vertices = np.array(pontos, dtype=np.float32)
    num_pontos_original = vertices.shape[0]
    print(f"Nuvem lida com {num_pontos_original:,} pontos válidos.")

    # =================================================================
    # PASSO NOVO 1: NORMALIZAÇÃO ESPACIAL (CRÍTICO PARA O PARSENET)
    # =================================================================
    print("A normalizar a escala do objeto para o espaço [-1.0, 1.0]...")
    # 1. Encontrar o centro de massa e transladar para o ponto (0,0,0)
    centro = np.mean(vertices, axis=0)
    vertices -= centro

    # 2. Encontrar a maior distância absoluta para encolher a peça proporcionalmente
    maior_distancia = np.max(np.abs(vertices))
    if maior_distancia > 0:
        vertices /= maior_distancia  # Agora a peça cabe perfeitamente na escala da IA

    # =================================================================
    # PASSO 2: DOWNSAMPLING INTELIGENTE (Substitui o random puro)
    # =================================================================
    MAX_POINTS = 50000
    if num_pontos_original > MAX_POINTS:
        print(f"A reduzir a nuvem para {MAX_POINTS:,} pontos de forma geométrica...")

        # Como estimar curvatura em NumPy puro sem bibliotecas pesadas pode ser lento para 3M,
        # usamos um Voxel Grid manual em NumPy que garante distribuição espacial perfeita:
        tamanho_voxel = 0.015 # Ajusta este valor se quiseres mais ou menos pontos

        # Agrupa os pontos em grelhas baseadas na sua nova posição (-1 a 1)
        coordenadas_voxel = np.floor(vertices / tamanho_voxel).astype(np.int32)
        _, indices_unicos = np.unique(coordenadas_voxel, axis=0, return_index=True)

        vertices_filtrados = vertices[indices_unicos]

        # Ajuste fino: Se o voxel grid der mais que 10k, refinamos. Se der menos, compensamos.
        if vertices_filtrados.shape[0] > MAX_POINTS:
            indices = np.random.choice(vertices_filtrados.shape[0], MAX_POINTS, replace=False)
            vertices = vertices_filtrados[indices]
        elif vertices_filtrados.shape[0] < MAX_POINTS:
            # Se o voxel foi agressivo demais, recheamos com pontos originais para não perder densidade
            pontos_em_falta = MAX_POINTS - vertices_filtrados.shape[0]
            indices_extra = np.random.choice(num_pontos_original, pontos_em_falta, replace=False)
            vertices = np.vstack((vertices_filtrados, vertices[indices_extra]))
        else:
            vertices = vertices_filtrados

    # Guarda o ficheiro leve, geométrico e normalizado
    os.makedirs("/content/parsenet-codebase/assets", exist_ok=True)
    np.savetxt(caminho_output, vertices, fmt='%.6f')
    print(f"Otimização concluída! Formato final dos vértices: {vertices.shape}")
    print(f"Limites reais pós-normalização: Min={vertices.min():.2f} / Max={vertices.max():.2f}\n")

else:
    print(f"Erro: O ficheiro '{args.input}' não foi encontrado.")