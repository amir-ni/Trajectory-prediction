import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import h3

def generate_embeddings(
    vocab: list,
    embedding_dim: int,
    mean: float = 0,
    std: float = 0.02,
    projection_matrix: np.ndarray = None,
    random_seed: int = 10,
) -> np.ndarray:
    """
    Generates embeddings for a given vocabulary based on axial coordinates.

    Args:
        vocab (list): A list of H3 hex addresses representing the vocabulary for which embeddings will be generated.
        embedding_dim (int): The dimension of the generated embedding vectors.
        mean (float, optional): The mean of the normal distribution used for embedding normalization. Default is 0.
        std (float, optional): The standard deviation of the normal distribution used for embedding normalization. Default is 0.02.
        projection_matrix (np.ndarray, optional): A 2D numpy array used for projecting the axial coordinates. If None, a random matrix will be generated.
    """
    origin_hex = vocab[0]
    base_i, base_j = h3.cell_to_local_ij(origin_hex, origin_hex)

    axial_coordinates = []
    for h3_hex in vocab:
        target_i, target_j = h3.cell_to_local_ij(origin_hex, h3_hex)
        q, r = target_i - base_i, target_j - base_j
        axial_coordinates.append((q,r))

    np.random.seed(random_seed)
    if projection_matrix is None:
        projection_matrix = np.random.randn(2, embedding_dim)

    projected_embedding = np.dot(axial_coordinates, projection_matrix)

    # standardized_embedding = (projected_embedding - np.mean(projected_embedding)) / np.std(projected_embedding)

    # normalized_embedding = standardized_embedding * std + mean

    eot_embedding = np.random.normal(loc=mean, scale=std, size=(1, embedding_dim))

    normal_samples = np.random.normal(loc=mean, scale=std, size=(len(vocab) * embedding_dim))
    flat_projected_embedding = projected_embedding.flatten()
    sorted_indices = np.argsort(flat_projected_embedding)
    sorted_projected_embedding = np.empty_like(flat_projected_embedding)
    sorted_projected_embedding[sorted_indices] = np.sort(normal_samples)
    sorted_projected_embedding = sorted_projected_embedding.reshape(projected_embedding.shape)

    return np.concatenate((eot_embedding, sorted_projected_embedding), axis=0)


def process_datasets(
    input_dir: Path,
    output_dir: Path,
    datasets_to_process: List[str],
    embedding_dim: Optional[int]
) -> None:
    """
    Process trajectory datasets for geolife, porto, and rome, generating vocab, mapping, neighbors,
    and transformed trajectory data.

    Args:
        input_dir (Path): Directory containing the input datasets.
        output_dir (Path): Directory where the processed data will be saved.
        datasets_to_process (List[str]): List of datasets to process (e.g., 'geolife', 'porto', 'rome').
    """
    datasets = {
        "geolife": [
            ("geolife7", input_dir / "geolife" / "ho_geolife_res7.csv", "date"),
            ("geolife8", input_dir / "geolife" / "ho_geolife_res8.csv", "date"),
            ("geolife9", input_dir / "geolife" / "ho_geolife_res9.csv", "date")
        ],
        "porto": [
            ("porto7", input_dir / "porto" / "ho_porto_res7.csv", "TIMESTAMP"),
            ("porto8", input_dir / "porto" / "ho_porto_res8.csv", "TIMESTAMP"),
            ("porto9", input_dir / "porto" / "ho_porto_res9.csv", "TIMESTAMP")
        ],
        "rome": [
            ("rome7", input_dir / "rome" / "ho_rome_res7.csv", "date"),
            ("rome8", input_dir / "rome" / "ho_rome_res8.csv", "date"),
            ("rome9", input_dir / "rome" / "ho_rome_res9.csv", "date")
        ]
    }

    for dataset_key in datasets_to_process:
        if dataset_key in datasets:
            for dataset in datasets[dataset_key]:
                dataset_name, file_path, date_column = dataset

                dataset_output_dir = output_dir / dataset_name
                dataset_output_dir.mkdir(parents=True, exist_ok=True)

                if not file_path.exists():
                    print(f"Warning: {file_path} does not exist. Skipping this dataset.")
                    continue

                df = pd.read_csv(file_path, header=0, usecols=["higher_order_trajectory", date_column],
                                 dtype={"higher_order_trajectory": "string", date_column: "string"})
                df = df.sort_values(by=[date_column])["higher_order_trajectory"].to_numpy()

                df_split = [i.split() for i in df]

                vocab = list(np.unique(np.concatenate(df_split, axis=0)))

                if embedding_dim is not None:
                    embeddings = generate_embeddings(vocab, embedding_dim)
                    embeddings_file_path = dataset_output_dir / 'embeddings.npy'
                    np.save(embeddings_file_path, embeddings)

                vocab = ["EOT"] + vocab
                vocab_file_path = dataset_output_dir / 'vocab.txt'
                with vocab_file_path.open('w', encoding='utf-8') as vocab_file:
                    vocab_file.write("\n".join(vocab) + "\n")

                mapping = {k: v for v, k in enumerate(vocab)}
                mapping_file_path = dataset_output_dir / 'mapping.json'
                with mapping_file_path.open('w', encoding='utf-8') as mapping_file:
                    json.dump(mapping, mapping_file, ensure_ascii=False)

                neighbors: Dict[int, List[int]] = dict()
                for x in vocab[1:]:
                    neighbors[mapping[str(x)]] = [mapping[i] for i in h3.grid_ring(str(x)) if i in vocab]
                neighbors_file_path = dataset_output_dir / 'neighbors.json'
                with neighbors_file_path.open('w', encoding='utf-8') as neighbors_file:
                    json.dump(neighbors, neighbors_file, ensure_ascii=False)

                df_mapped = [[str(mapping[j]) for j in i] for i in df_split]
                data_file_path = dataset_output_dir / 'data.txt'
                with data_file_path.open('w', encoding='utf-8') as data_file:
                    for item in df_mapped:
                        data_file.write(' '.join(item) + f" {mapping['EOT']}\n")

                print(f"Processing completed for {dataset_name}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Prediction Learning for geolife, porto, and rome datasets')

    parser.add_argument('--input_dir', type=Path, default=Path('data'),
                        help='Path to input dataset files (default: ./data)')

    parser.add_argument('--output_dir', type=Path, default=Path('data'),
                        help='Path to output directory (default: ./data)')

    parser.add_argument('--datasets', type=str, nargs='+', choices=['geolife', 'porto', 'rome'], required=True,
                        help='Specify which datasets to process (choose from geolife, porto, rome)')

    parser.add_argument('--embedding_dim', type=int,
                        help="Dimension of the generated embedding vectors. If not provided, embeddings will not be generated.")

    args = parser.parse_args()

    process_datasets(args.input_dir, args.output_dir, args.datasets, args.embedding_dim)
