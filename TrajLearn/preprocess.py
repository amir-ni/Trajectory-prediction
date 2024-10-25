import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import h3
from typing import List, Dict

def process_datasets(input_dir: Path, output_dir: Path, datasets_to_process: List[str]) -> None:
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
            ("geolife-7", input_dir / "geolife" / "ho_geolife_res7.csv", "date"),
            ("geolife-8", input_dir / "geolife" / "ho_geolife_res8.csv", "date"),
            ("geolife-9", input_dir / "geolife" / "ho_geolife_res9.csv", "date")
        ],
        "porto": [
            ("porto-7", input_dir / "porto" / "ho_porto_res7.csv", "TIMESTAMP"),
            ("porto-8", input_dir / "porto" / "ho_porto_res8.csv", "TIMESTAMP"),
            ("porto-9", input_dir / "porto" / "ho_porto_res9.csv", "TIMESTAMP")
        ],
        "rome": [
            ("rome-7", input_dir / "rome" / "ho_rome_res7.csv", "date"),
            ("rome-8", input_dir / "rome" / "ho_rome_res8.csv", "date"),
            ("rome-9", input_dir / "rome" / "ho_rome_res9.csv", "date")
        ]
    }

    for dataset_key in datasets_to_process:
        if dataset_key in datasets:
            for dataset in datasets[dataset_key]:
                dataset_name, file_path, date_column = dataset

                # Create output directory for this dataset
                dataset_output_dir = output_dir / dataset_name
                dataset_output_dir.mkdir(parents=True, exist_ok=True)

                # Check if the file exists before processing
                if not file_path.exists():
                    print(f"Warning: {file_path} does not exist. Skipping this dataset.")
                    continue

                # Load the CSV file
                df = pd.read_csv(file_path, header=0, usecols=["higher_order_trajectory", date_column],
                                 dtype={"higher_order_trajectory": "string", date_column: "string"})
                df = df.sort_values(by=[date_column])["higher_order_trajectory"].to_numpy()

                # Split trajectory data
                df_split = [i.split() for i in df]

                # Create vocabulary list and save to file
                vocab = ["EOT"] + list(np.unique(np.concatenate(df_split, axis=0)))
                vocab_file_path = dataset_output_dir / 'vocab.txt'
                with vocab_file_path.open('w', encoding='utf-8') as vocab_file:
                    vocab_file.write("\n".join(vocab) + "\n")

                # Create mapping and save to JSON
                mapping = {k: v for v, k in enumerate(vocab)}
                mapping_file_path = dataset_output_dir / 'mapping.json'
                with mapping_file_path.open('w', encoding='utf-8') as mapping_file:
                    json.dump(mapping, mapping_file, ensure_ascii=False)

                # Create neighbors for each hex and save to JSON
                neighbors: Dict[int, List[int]] = dict()
                for x in vocab[1:]:  # Skip 'EOT'
                    neighbors[mapping[str(x)]] = [mapping[i] for i in h3.hex_ring(str(x)) if i in vocab]
                neighbors_file_path = dataset_output_dir / 'neighbors.json'
                with neighbors_file_path.open('w', encoding='utf-8') as neighbors_file:
                    json.dump(neighbors, neighbors_file, ensure_ascii=False)

                # Map trajectories to their respective indices and save to file
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

    args = parser.parse_args()

    process_datasets(args.input_dir, args.output_dir, args.datasets)
