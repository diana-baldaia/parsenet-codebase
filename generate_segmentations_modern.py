import argparse
import numpy as np
import torch
import os

from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.mean_shift import MeanShift

# =====================================================================
# 1. NORMALIZAÇÃO ESPACIAL ISOMÉTRICA (SEM ROTAÇÃO PCA)
# Centra a nuvem no zero e escala uniformemente com base no maior eixo.
# Isto garante que a grelha mantém a sua orientação original (ortogonal).
# =====================================================================
def normalize_points(points):
    EPS = np.finfo(np.float32).eps
    # Centrar na origem (0, 0, 0)
    points = points - np.mean(points, 0, keepdims=True)
    
    # Escala isométrica pura (idêntica à do script de downsampling)
    maior_distancia = np.max(np.abs(points))
    if maior_distancia > 0:
        points = points / (maior_distancia + EPS)
        
    return points.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="abc_00470_downs.xyz", help="Caminho do ficheiro .xyz gerado pelo downsampling")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando no dispositivo: {device}")

    path_in = args.input
    if not os.path.exists(path_in):
        print(f"Erro: ficheiro '{path_in}' não encontrado. Execução terminada.")
        exit(1)

    # =====================================================================
    # 2. CONFIGURAÇÃO ATIVA: MODO COM NORMAIS (6 CANAIS)
    # =====================================================================
    USAR_NORMAIS = True 
    
    if USAR_NORMAIS:
        pth_path = "./logs/results/parsenet.pth/parsenet.pth"  # O teu modelo completo com normais
        canais = 6
        print("-> Modo: 6 Canais (XYZ + Normais de Superfície) ATIVADO.")
    else:
        pth_path = "./logs/results/parsenet_no_normals.pth/parsenet_no_normals.pth"
        canais = 3
        print("-> Modo: 3 Canais (Apenas XYZ) ATIVADO.")

    # Inicializar a arquitetura DGCN do ParseNet
    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        loss_function=None,
        mode=0,
        num_channels=canais,  # Definido dinamicamente (3 ou 6)
    )
    model.to(device)
    model.eval()

    # Carregar os pesos do checkpoint parsenet.pth
    print(f"A carregar pesos de: {pth_path} ...")
    checkpoint = torch.load(pth_path, map_location=device)
    
    # Limpar o prefixo 'module.' caso tenha sido gravado com DataParallel
    clean_checkpoint = {}
    for k, v in checkpoint.items():
        clean_key = k[len("module."):] if k.startswith("module.") else k
        clean_checkpoint[clean_key] = v

    missing, unexpected = model.load_state_dict(clean_checkpoint, strict=False)
    if missing:
        print(f"Aviso - Chaves em falta no checkpoint: {missing}")
    print("Pesos do modelo carregados com sucesso!")

    # =====================================================================
    # 3. CARREGAMENTO E PREPARAÇÃO DOS DADOS (XYZ + NORMAIS)
    # =====================================================================
    print(f"A ler o ficheiro {path_in}...")
    dados_brutos = np.loadtxt(path_in).astype(np.float32)
    
    # Extrair as coordenadas espaciais (Primeiras 3 colunas)
    pontos_xyz = dados_brutos[:, :3]
    
    # Normalizar estritamente a posição da geometria
    pontos_norm = normalize_points(pontos_xyz)
    
    if USAR_NORMAIS:
        if dados_brutos.shape[1] >= 6:
            # Extrair as normais geradas pelo Open3D (Colunas 4, 5 e 6)
            normais = dados_brutos[:, 3:6]
            # Combinar a geometria normalizada com as normais originais
            pontos_input = np.hstack((pontos_norm, normais))
            print("-> Dados preparados com sucesso: Matriz unida [XYZ_norm + NxNyNz].")
        else:
            print("ERRO CRÍTICO: O ficheiro de entrada não contém as colunas das normais!")
            print("Certifica-te de que rodaste o script de downsampling com Open3D primeiro.")
            exit(1)
    else:
        pontos_input = pontos_norm

    # Transformar em Tensor do PyTorch com dimensão de Batch: (1, N, C)
    points_tensor = torch.from_numpy(pontos_input)[None, :].to(device)

    # =====================================================================
    # 4. INFERÊNCIA DA REDE E EXTRAÇÃO DE EMBEDDINGS
    # =====================================================================
    with torch.no_grad():
        # Criar labels fictícios exigidos pela assinatura do modelo
        dummy_labels = torch.zeros(1, points_tensor.shape[1], dtype=torch.long).to(device)
        
        # Permutar para o formato correto do PointNet (Batch, Canais, Número_Pontos) -> (1, 6, N)
        embedding, _, _ = model(points_tensor.permute(0, 2, 1), dummy_labels, False)

    # Normalizar os embeddings no espaço esférico L2
    embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)

    # =====================================================================
    # 5. AGRUPAMENTO COM MEANSHIFT AJUSTADO
    # =====================================================================
    # Ajuste do Quantile: Como agora temos normais, a rede terá certezas muito
    # maiores sobre as transições de planos. 
    # - Se notar retalhos pequenos: aumenta para 0.035 ou 0.040.
    # - Se a grelha se fundir toda numa única cor: reduz para 0.020.
    QUANTILE_ALVO = 0.01
    print(f"A executar o MeanShift Clustering (Quantile: {QUANTILE_ALVO})...")
    
    ms = MeanShift()
    _, _, cluster_ids = ms.guard_mean_shift(embedding, quantile=QUANTILE_ALVO, iterations=10)

    # =====================================================================
    # 6. EXPORTAÇÃO DO FICHEIRO XYZC (Pronto para o CloudCompare/MeshLab)
    # =====================================================================
    base_name = os.path.splitext(os.path.basename(path_in))[0]
    output_path = os.path.join(f"{base_name}_segmented.xyzc")
    
    # Guardamos os pontos XYZ na sua escala original, acrescentando o ID do cluster no fim
    classes_numpy = cluster_ids.data.cpu().numpy()[:, None]
    dados_saida = np.hstack([pontos_xyz, classes_numpy])
    
    np.savetxt(output_path, dados_saida, fmt='%.6f')
    
    num_segmentos = len(np.unique(classes_numpy))
    print(f"\n🚀 SUCESSO! Segmentação concluída com sucesso.")
    print(f"Ficheiro guardado em: {output_path}")
    print(f"Número total de primitivas geométricas detetadas: {num_segmentos}")