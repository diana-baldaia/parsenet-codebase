import argparse
import numpy as np
import torch
import os

from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.mean_shift import MeanShift

# --- Utilitários de Geometria ---
def pca_numpy(points):
    covariance_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors

def rotation_matrix_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-6:
        return np.eye(3)
    v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s ** 2))
    return R

def normalize_points(points):
    EPS = np.finfo(np.float32).eps
    points = points - np.mean(points, 0, keepdims=True)
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points = (R @ points.T).T
    std = np.max(points, 0) - np.min(points, 0)
    points = points / (np.max(std) + EPS)
    return points.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="abc_00470.xyz", help="Nome do ficheiro .xyz dentro da pasta assets")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando no dispositivo: {device}")

    path_in = os.path.join("./assets", args.input)
    if not os.path.exists(path_in):
        print(f"Erro: ficheiro '{path_in}' não encontrado. Execução terminada.")
        exit(1)

    pth_path = "./logs/results/parsenet_no_normals.pth/parsenet_no_normals.pth"

    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        loss_function=None,
        mode=0,
        num_channels=3,
    )
    model.to(device)
    model.eval()

    checkpoint = torch.load(pth_path, map_location=device)

    # Strip 'module.' prefix from DataParallel-saved weights
    clean_checkpoint = {}
    for k, v in checkpoint.items():
        clean_key = k[len("module."):] if k.startswith("module.") else k
        clean_checkpoint[clean_key] = v

    missing, unexpected = model.load_state_dict(clean_checkpoint, strict=False)
    if missing:
        print(f"Chaves em falta no checkpoint: {missing}")
    if unexpected:
        print(f"Chaves extra no checkpoint (ignoradas): {unexpected}")
    print("Pesos do modelo carregados com sucesso!")

    points = np.loadtxt(path_in).astype(np.float32)
    points_norm = normalize_points(points)
    points_tensor = torch.from_numpy(points_norm)[None, :].to(device)

    with torch.no_grad():
        dummy_labels = torch.zeros(1, points_tensor.shape[1], dtype=torch.long).to(device)
        embedding, _, _ = model(points_tensor.permute(0, 2, 1), dummy_labels, False)

    embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)
    ms = MeanShift()
    _, _, cluster_ids = ms.guard_mean_shift(embedding, quantile=0.015, iterations=10)

    base_name = os.path.splitext(os.path.basename(path_in))[0]
    output_path = os.path.join("./assets", f"{base_name}_segmented.xyzc")
    np.savetxt(
        output_path,
        np.hstack([points, cluster_ids.data.cpu().numpy()[:, None]]),
    )
    print(f"\nSUCESSO! Predição salva em: {output_path}")
    print(f"Número de segmentos encontrados: {len(np.unique(cluster_ids.data.cpu().numpy()))}")
