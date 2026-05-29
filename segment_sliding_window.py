import os
import argparse
import numpy as np
import torch
from src.PointNet import PrimitivesEmbeddingDGCNGn
from sklearn.cluster import DBSCAN

def normalize_global_points(points):
    """Normaliza a nuvem inteira para o espaço [-1, 1] antes do loop"""
    EPS = np.finfo(np.float32).eps
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.abs(points_centered))
    if max_dist > 0:
        points_centered /= (max_dist + EPS)
    return points_centered, centroid, max_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Caminho da nuvem (gerada com ~100k-500k pontos)")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando Sliding Window Otimizado em: {device}")

    # 1. Carregar os Dados
    raw_data = np.loadtxt(args.input).astype(np.float32)
    xyz_original = raw_data[:, :3]
    tem_normais = raw_data.shape[1] >= 6
    normais_global = raw_data[:, 3:6] if tem_normais else None

    # 2. NORMALIZAÇÃO PREVENTIVA (Garante que a escala está entre -1 e 1)
    print("A normalizar escala global para otimização do loop...")
    xyz_global, centro_original, escala_original = normalize_global_points(xyz_original)

    # 3. Configurar Modelo
    canais = 6 if tem_normais else 3
    pth_path = "./logs/results/parsenet.pth/parsenet.pth" if tem_normais else "./logs/results/parsenet_no_normals.pth/parsenet_no_normals.pth"
    
    model = PrimitivesEmbeddingDGCNGn(embedding=True, emb_size=128, primitives=True, num_primitives=10, loss_function=None, mode=0, num_channels=canais)
    model.to(device).eval()
    
    checkpoint = torch.load(pth_path, map_location=device)
    clean_checkpoint = {k[len("module."):] if k.startswith("module.") else k: v for k, v in checkpoint.items()}
    model.load_state_dict(clean_checkpoint, strict=False)
    print("Pesos carregados com sucesso!")

    # =====================================================================
    # 4. PARÂMETROS DA JANELA (AGORA FIXOS NO ESPAÇO NORMALIZADO)
    # =====================================================================
    # Como a peça agora mede no máximo 2 unidades (de -1 a 1):
    tamanho_janela = 0.4  # Cubo que inspeciona 20% da peça de cada vez
    passo = 0.20          # 50% de sobreposição

    min_bounds = xyz_global.min(axis=0)
    max_bounds = xyz_global.max(axis=0)

    todos_embeddings = np.zeros((xyz_global.shape[0], 128), dtype=np.float32)
    contagem_votos = np.zeros(xyz_global.shape[0], dtype=np.int32)

    xs = np.arange(min_bounds[0], max_bounds[0], passo)
    ys = np.arange(min_bounds[1], max_bounds[1], passo)
    zs = np.arange(min_bounds[2], max_bounds[2], passo)

    total_janelas_teoricas = len(xs) * len(ys) * len(zs)
    print(f"Grelha gerada: {len(xs)}x{len(ys)}x{len(zs)} = {total_janelas_teoricas} janelas a analisar.")

    janelas_processadas = 0
    for x in xs:
        for y in ys:
            for z in zs:
                # Filtragem ultra-rápida usando máscaras booleanas
                indices_bloco = np.where(
                    (xyz_global[:, 0] >= x) & (xyz_global[:, 0] < x + tamanho_janela) &
                    (xyz_global[:, 1] >= y) & (xyz_global[:, 1] < y + tamanho_janela) &
                    (xyz_global[:, 2] >= z) & (xyz_global[:, 2] < z + tamanho_janela)
                )[0]

                if len(indices_bloco) < 100:  # Ignora o vazio absoluto
                    continue

                janelas_processadas += 1
                if janelas_processadas % 10 == 0:
                    print(f"-> Janelas geométricas ativas processadas: {janelas_processadas}...")

                # Se o bloco for muito denso, limitamos para o PointNet não estourar a memória
                if len(indices_bloco) > 15000:
                    idx_sub = np.random.choice(len(indices_bloco), 15000, replace=False)
                    indices_bloco = indices_bloco[idx_sub]

                xyz_local = xyz_global[indices_bloco]
                
                # Deslocamento local interno para centralização fina exigida pelo dataset ABC
                xyz_local_centrado = xyz_local - np.mean(xyz_local, axis=0)

                if tem_normais:
                    input_local = np.hstack((xyz_local_centrado, normais_global[indices_bloco]))
                else:
                    input_local = xyz_local_centrado

                tensor_input = torch.from_numpy(input_local)[None, :].to(device)
                with torch.no_grad():
                    dummy_labels = torch.zeros(1, tensor_input.shape[1], dtype=torch.long).to(device)
                    emb_local, _, _ = model(tensor_input.permute(0, 2, 1), dummy_labels, False)
                
                emb_local = torch.nn.functional.normalize(emb_local[0].T, p=2, dim=1).cpu().numpy()

                todos_embeddings[indices_bloco] += emb_local
                contagem_votos[indices_bloco] += 1

    print(f"Processamento de janelas concluído. Total de janelas úteis: {janelas_processadas}")
    contagem_votos[contagem_votos == 0] = 1
    embeddings_globais = todos_embeddings / contagem_votos[:, None]

    # =====================================================================
    # 5. CLUSTERING ROBUSTO E AUTOMÁTICO (DBSCAN ADAPTATIVO + LIMPEZA)
    # =====================================================================
    print("A executar agrupamento geométrico cego (DBSCAN)...")
    
    # eps=0.35 dá o balanço ideal para os embeddings gerados por janelas locais
    # min_samples exige que uma primitiva tenha consistência mínima de pontos
    db = DBSCAN(eps=0.35, min_samples=30, metric='euclidean', n_jobs=-1)
    cluster_ids_brutos = db.fit_predict(embeddings_globais)
    
    # --- PÓS-PROCESSAMENTO INTELIGENTE PARA ELIMINAR MICRO-RETALHOS ---
    print("A limpar micro-segmentos ruidosos automaticamente...")
    cluster_ids_limpos = cluster_ids_brutos.copy()
    
    # Contar quantos pontos cada segmento tem
    ids_unicos, contagens = np.unique(cluster_ids_brutos, return_counts=True)
    
    # Definimos um limiar dinâmico: qualquer primitiva com menos de 1.5% 
    # dos pontos totais da peça é considerada ruído de transição da janela
    limiar_ruido = int(0.015 * xyz_global.shape[0]) 
    
    # Identificar os clusters que são válidos e os que são pequenos retalhos
    clusters_ruidosos = ids_unicos[contagens < limiar_ruido]
    clusters_validos = ids_unicos[contagens >= limiar_ruido]
    # Remover o ID -1 (pontos que o DBSCAN já considerou ruído bruto) dos válidos
    clusters_validos = clusters_validos[clusters_validos != -1]
    
    if len(clusters_ruidosos) > 0 and len(clusters_validos) > 0:
        # Criar uma máscara para os pontos que pertencem a segmentos pequenos
        mascara_ruido = np.isin(cluster_ids_limpos, clusters_ruidosos) | (cluster_ids_limpos == -1)
        indices_ruido = np.where(mascara_ruido)[0]
        indices_validos = np.where(~mascara_ruido)[0]
        
        if len(indices_validos) > 0:
            # Reatribuir os micro-segmentos ao segmento válido mais próximo no espaço geométrico 3D
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
            knn.fit(xyz_global[indices_validos], cluster_ids_limpos[indices_validos])
            
            valores_corrigidos = knn.predict(xyz_global[indices_ruido])
            cluster_ids_limpos[indices_ruido] = valores_corrigidos

    # Re-mapear os IDs para ficarem ordenados consecutivamente de 0 a N
    _, indices_mapeamento = np.unique(cluster_ids_limpos, return_inverse=True)
    cluster_ids_finais = indices_mapeamento

    # =====================================================================
    # 6. VOLTAR À ESCALA REAL ORIGINAL E SALVAR
    # =====================================================================
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = f"{base_name}_auto_segmented.xyzc"
    
    np.savetxt(output_path, np.hstack([xyz_original, cluster_ids_finais[:, None]]), fmt='%.6f')
    
    num_segmentos_reais = len(np.unique(cluster_ids_finais))
    print(f"\n SUCESSO! Algoritmo adaptou-se e extraiu autonomamente a geometria.")
    print(f"Ficheiro guardado em: {output_path}")
    print(f"Número de planos robustos isolados: {num_segmentos_reais}")