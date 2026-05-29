import os
import argparse
import numpy as np

# Se não tiveres o open3d no Colab, basta correr: !pip install open3d
try:
    import open3d as o3d
except ImportError:
    print("Por favor, instala o open3d: pip install open3d")
    raise

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Caminho do ficheiro .xyz de entrada")
args, _ = parser.parse_known_args()

caminho_original = args.input
caminho_output = os.path.splitext(args.input)[0] + "_downs.xyz"

if os.path.exists(caminho_original):
    print(f"A abrir o ficheiro gigante {args.input}...")

    # Carregar pontos usando o Open3D (muito mais rápido que loops em Python nativo)
    pcd = o3d.io.read_point_cloud(caminho_original, format='xyz')
    num_pontos_original = len(pcd.points)
    print(f"Nuvem lida com {num_pontos_original:,} pontos válidos.")

    # =================================================================
    # PASSO 1: DOWNSAMPLING GEOMÉTRICO (Voxel robusto via Open3D)
    # =================================================================
    MAX_POINTS = 20000  # O limite padrão do ParseNet costuma ser 10k ou 40k, ajusta conforme o teu modelo
    print(f"A aplicar Voxel Downsampling para preservar a estrutura da grelha...")
    
    # Ajusta o tamanho do voxel dinamicamente ou usa um valor fixo adequado à escala original
    # Para estruturas finas, queremos um voxel pequeno.
    tamanho_voxel = 0.5  # <--- Altera isto dependendo da escala original do teu scanner
    pcd_filtrado = pcd.voxel_down_sample(voxel_size=tamanho_voxel)
    
    # Ajuste fino rigoroso para garantir EXACTAMENTE o número de pontos que a rede quer
    pontos_filtrados = np.asarray(pcd_filtrado.points)
    
    if len(pontos_filtrados) > MAX_POINTS:
        # Farthest Point Sampling (FPS) seria o ideal, mas um choice aleatório nos voxels já é melhor do que na nuvem bruta
        indices = np.random.choice(len(pontos_filtrados), MAX_POINTS, replace=False)
        pcd_final = pcd_filtrado.select_by_index(indices)
    elif len(pontos_filtrados) < MAX_POINTS:
        print(f"Aviso: O voxel size gerou apenas {len(pontos_filtrados)} pontos. Usando amostragem aleatória uniforme para compensar...")
        indices = np.random.choice(num_pontos_original, MAX_POINTS, replace=False)
        pcd_final = pcd.select_by_index(indices)
    else:
        pcd_final = pcd_filtrado

    # =================================================================
    # PASSO 2: CÁLCULO DE NORMAIS (Crucial para o ParseNet)
    # =================================================================
    print("A estimar normais de superfície...")
    # Estima as normais olhando para os 30 vizinhos mais próximos de cada ponto
    pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    pcd_final.orient_normals_consistent_tangent_plane(k=15)

    vertices = np.asarray(pcd_final.points)
    normais = np.asarray(pcd_final.normals)

    # =================================================================
    # PASSO 3: NORMALIZAÇÃO ESPACIAL ISOMÉTRICA
    # =================================================================
    print("A normalizar a escala para o espaço [-1.0, 1.0]...")
    centro = np.mean(vertices, axis=0)
    vertices -= centro

    maior_distancia = np.max(np.abs(vertices))
    if maior_distancia > 0:
        vertices /= maior_distancia

    # Juntar as coordenadas com as normais [X, Y, Z, Nx, Ny, Nz]
    dados_finais = np.hstack((vertices, normais))

    # =================================================================
    # GUARDAR FICHEIRO FINAL
    # =================================================================
    os.makedirs("/content/parsenet-codebase/assets", exist_ok=True)
    # Guardar com as 6 colunas que o ParseNet espera
    np.savetxt(caminho_output, dados_finais, fmt='%.6f')
    print(f"Otimização concluída! Formato final da matriz: {dados_finais.shape}")
    print(f"Limites reais pós-normalização: Min={vertices.min():.2f} / Max={vertices.max():.2f}\n")

else:
    print(f"Erro: O ficheiro '{args.input}' não foi encontrado.")