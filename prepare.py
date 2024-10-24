import os
import h3
import json
import argparse
import numpy as np
import pandas as pd

# Main processing function
def process_datasets(input_dir, output_dir, datasets_to_process):
    # Define the datasets we want to handle (geolife, porto, rome)
    datasets = {
        "geolife": [
            ("geolife-7", os.path.join(input_dir, "ho_geolife", "ho_geolife_res7.csv"), "date"),
            ("geolife-8", os.path.join(input_dir, "ho_geolife", "ho_geolife_res8.csv"), "date"),
            ("geolife-9", os.path.join(input_dir, "ho_geolife", "ho_geolife_res9.csv"), "date")
        ],
        "porto": [
            ("porto-7", os.path.join(input_dir, "ho_porto", "ho_porto_res7.csv"), "TIMESTAMP"),
            ("porto-8", os.path.join(input_dir, "ho_porto", "ho_porto_res8.csv"), "TIMESTAMP"),
            ("porto-9", os.path.join(input_dir, "ho_porto", "ho_porto_res9.csv"), "TIMESTAMP")
        ],
        "rome": [
            ("rome-7", os.path.join(input_dir, "ho_rome", "ho_rome_res7.csv"), "date"),
            ("rome-8", os.path.join(input_dir, "ho_rome", "ho_rome_res8.csv"), "date"),
            ("rome-9", os.path.join(input_dir, "ho_rome", "ho_rome_res9.csv"), "date")
        ]
    }

    # Process the selected datasets
    for dataset_key in datasets_to_process:
        if dataset_key in datasets:
            for dataset in datasets[dataset_key]:
                dataset_name, file_path, date_column = dataset

                # Create output directory for this dataset
                dataset_output_dir = os.path.join(output_dir, dataset_name)
                os.makedirs(dataset_output_dir, exist_ok=True)

                # Check if the file exists before processing
                if not os.path.exists(file_path):
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
                with open(os.path.join(dataset_output_dir, 'vocab.txt'), 'w') as vocab_file:
                    for item in vocab:
                        vocab_file.write(item + "\n")

                # Create mapping and save to JSON
                mapping = {k: v for v, k in enumerate(vocab)}
                with open(os.path.join(dataset_output_dir, 'mapping.json'), 'w') as mapping_file:
                    mapping_file.write(json.dumps(mapping))

                # Create neighbors for each hex and save to JSON
                neighbors = dict()
                for x in vocab[1:]:  # Skip 'EOT'
                    neighbors[mapping[str(x)]] = [mapping[i] for i in h3.hex_ring(str(x)) if i in vocab]
                with open(os.path.join(dataset_output_dir, 'neighbors.json'), 'w') as neighbors_file:
                    neighbors_file.write(json.dumps(neighbors))

                # Map trajectories to their respective indices and save to file
                df_mapped = [[str(mapping[j]) for j in i] for i in df_split]
                with open(os.path.join(dataset_output_dir, 'data.txt'), 'w') as data_file:
                    for item in df_mapped:
                        data_file.write(' '.join(item) + " " + str(mapping['EOT']) + "\n")

                print(f"Processing completed for {dataset_name}.")

if __name__ == '__main__':
    # Argument parser for input and output directories and datasets
    parser = argparse.ArgumentParser(description='Trajectory Prediction Learning for geolife, porto, and rome datasets')
    
    # Optional input_dir argument (defaults to "./data" in the current directory)
    parser.add_argument('--input_dir', type=str, default="./data", 
                        help='Path to input dataset files (default: ./data)')
    
    # Optional output_dir argument (defaults to "./data" in the current directory)
    parser.add_argument('--output_dir', type=str, default="./data", 
                        help='Path to output directory (default: ./data)')
    
    # Required argument to specify which datasets to process
    parser.add_argument('--datasets', type=str, nargs='+', choices=['geolife', 'porto', 'rome'], required=True,
                        help='Specify which datasets to process (choose from geolife, porto, rome)')

    # Parse the arguments
    args = parser.parse_args()

    # Run the processing function with the provided arguments
    process_datasets(args.input_dir, args.output_dir, args.datasets)
