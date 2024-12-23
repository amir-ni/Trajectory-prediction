import os
import ast
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from collections.abc import Callable

import h3
import geopandas
import matplotlib
import numpy as np
import pandas as pd
import contextily as cx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def gps_to_h3(
    gps_points: list,
    resolution: int,
    boundary: dict,
):
    """
    Convert a list of GPS coordinates to H3 hexagons at a specified resolution.

    Arguments:
    - gps_points: list of tuples, where each tuple contains (longitude, latitude) in degrees.
                  Example: [(lon1, lat1), (lon2, lat2), ...]
    - resolution: int, the H3 resolution level to use for converting coordinates.
                  Higher resolutions create smaller hexagons.

    Returns:
    - A list of H3 hexagon IDs corresponding to the input GPS coordinates.
    """
    result = []
    for lon, lat in gps_points:
        if boundary["Min_lon"] <= lon and boundary["Min_lat"] <= lat and \
            boundary["Max_lon"] >= lon and boundary["Max_lat"] >= lat:
            cell = h3.latlng_to_cell(lat, lon, resolution)
            if len(result) == 0 or cell != result[-1]:
                result.append(h3.latlng_to_cell(lat, lon, resolution))
    return result


def preprocess_resolution(
    dataset: str = 'geolife',
    min_resolution: int = 7,
    max_resolution: int = 9,
    output_dir: str = 'data/mixed_res',
    save_csv: bool = False,
    use_boundary: bool = True,
):
    """
    Preprocess a trajectory dataset by converting GPS points to H3 hexagon representations
    at various resolutions, and save the results and associated data structures.

    This function reads a CSV file containing GPS points, converts these points to H3
    hexagon IDs at specified resolutions, and optionally saves the results as CSV and
    pickle files. It also calculates hexagon counts and neighbor relationships.

    Arguments:
    - dataset: str, name of the dataset to process (e.g., 'geolife', 'rome', 'porto').
    - min_resolution: int, the minimum H3 resolution level for hexagon generation.
    - max_resolution: int, the maximum H3 resolution level for hexagon generation.
    - output_dir: str, the directory where processed data and outputs will be saved.
    - save_csv: bool, if True, saves processed data and hexagon counts as CSV files.

    Steps Performed:
    1. Reads GPS points from a dataset CSV file and applies a transformation to generate
       H3 hexagon IDs at each resolution from `min_resolution` to `max_resolution`.
    2. Saves the transformed dataset to a CSV and pickle file if `save_csv` is True.
    3. Computes and saves hexagon counts and unique hexagon IDs at each resolution.
    4. Computes neighbor relationships for hexagons and saves them as a pickle file.

    Outputs:
    - Processed dataset with hexagon columns at specified resolutions (CSV and/or pickle).
    - Hexagon count file, detailing how many times each hexagon appears (CSV and pickle).
    - A dictionary of unique hexagons at each resolution (pickle).
    - Neighbor relationships for each hexagon, excluding child hexagons (pickle).
    """
    os.makedirs(output_dir, exist_ok=True)

    boundary = {
        'geolife':{
            "Min_lat":39.50000000,
            "Max_lat":40.50000000,
            "Min_lon":116.00000000,
            "Max_lon":117.00000000
        },
        'rome':{
            "Min_lat":41.793710,
            "Max_lat":41.991390,
            "Min_lon":12.372598,
            "Max_lon":12.622537
        },
        'porto': {
            "Min_lat": -1000,
            "Max_lat": 1000,
            "Min_lon": -1000,
            "Max_lon": 1000
        }
    }

    data_path = {
        'geolife':'data/raw_aggrigated/geolife_aggregated.csv',
        'rome': 'data/raw_aggrigated/rome_taxi_aggregated.csv',
        'porto': 'data/raw_aggrigated/porto.csv'
    }

    file_path = data_path[dataset]
    print(f"Processing file: {file_path}")

    target_column = 'route_points' if 'geolife' in file_path or 'rome_taxi' in file_path else 'TIMESTAMP'
    target_time = 'date' if 'geolife' in file_path or 'rome_taxi' in file_path else 'POLYLINE'
    df = pd.read_csv(file_path, usecols=[target_column, target_time]).rename(columns={target_column: 'points', target_time: 'time'})

    df['points'] = df['points'].apply(ast.literal_eval)
    print("Processed points")
    for resolution in range(min_resolution, max_resolution+1):
        column_name = f'hex_{resolution}'
        if use_boundary:
            df[column_name] = df['points'].apply(gps_to_h3, args=(resolution, boundary[dataset],))
        else:
            df[column_name] = df['points'].apply(gps_to_h3, args=(resolution,))
        print(f"Processed resolution: {resolution}")

    if save_csv:
        df.to_csv(f"{output_dir}/{dataset}.csv", index=False)
        print("Saved dataset csv")

    with open(f'{output_dir}/{dataset}.pkl', 'wb') as f:
        pickle.dump(df, f)
    print("Saved dataset pickle")

    hex_counts = defaultdict(int)
    hexes = {res: set() for res in range(min_resolution, max_resolution + 1)}

    for res in range(min_resolution, max_resolution + 1):
        column_name = f'hex_{res}'
        for hex_list in df[column_name]:
            for hex_id in hex_list:
                hex_counts[hex_id] += 1
                hexes[res].add(hex_id)

    if save_csv:
        hex_df = pd.DataFrame.from_dict(hex_counts, orient='index', columns=['occurrences'])
        hex_df.index.name = 'hex_id'
        hex_df.to_csv(f'{output_dir}/hex_count_{dataset}.csv')
        print("Saved hexagon count csv")


    with open(f'{output_dir}/hexes_{dataset}.pkl', 'wb') as f:
        pickle.dump(hexes, f)
    print("Saved hexagons dictionary pickle")


    with open(f'{output_dir}/hex_count_{dataset}.pkl', 'wb') as f:
        pickle.dump(hex_counts, f)
    print("Saved hexagon count pickle")

    neighbors = defaultdict(set)
    for resolution in range(min_resolution, max_resolution+1):
        for hex_id in hexes[resolution]:
            hex_neighbors = set()
            hex_children = set()

            for children_resolution in range(resolution+1, max_resolution+1):
                hex_children.update(h3.cell_to_children(hex_id, children_resolution))

            for hex_child in hex_children:
                hex_neighbors.update(h3.grid_ring(hex_child, 1))
            neighbors[hex_id] = hex_neighbors - hex_children

    for hex_id, neighbors_set in list(neighbors.items()):
        for neighbor in neighbors_set:
            neighbors[neighbor].add(hex_id)

    with open(f'{output_dir}/neighbors_{dataset}.pkl', 'wb') as f:
        pickle.dump(neighbors, f)
    print("Saved hexagon neighbors pickle")


def visualize(hex_counts: dict, output_path: str, bins:list = None, **kwargs):
    """
    Generate and display a heatmap of hexagon counts on a map, and save it as an image.

    Arguments:
    - hex_seq: list of str, a list of hexagon sequences.
    - zoom_level: int, the zoom level for the map.
    - output_dir: str, the directory path to save any output if necessary.
    - **kwargs: Options to pass to geopandas plotting method.

    The function flattens the input hex sequences, counts the occurrences of each hexagon,
    creates a GeoJSON object, and visualizes it using Plotly with a heatmap.
    """
    matplotlib.rcParams.update({'font.size': 28})
    df_hex_plot = pd.DataFrame(hex_counts.items(), columns=['hex_id', 'count'])

    df_hex_plot['geometry'] = df_hex_plot['hex_id'].apply(lambda x: h3.cells_to_h3shape([x]))
    df = geopandas.GeoDataFrame(df_hex_plot.drop(columns=['hex_id']), crs='EPSG:4326')
    df = df.to_crs(epsg=3857)

    _, ax = plt.subplots(figsize=(24,24))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if bins is not None:
        labels = ['Low', 'Medium', 'High']
        df['count'] = pd.cut(df['count'], bins=bins, labels=labels, include_lowest=True)

    df.plot(
        ax=ax,
        alpha=0.9,
        edgecolor=(134/256, 218/256, 227/256, 0.1),
        linewidth=0.001,
        column='count',
        legend=True,
        **kwargs,
    )

    cx.add_basemap(ax, crs=df.crs, source=cx.providers.CartoDB.Positron)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches='tight')


def threshold_split_condition(threshold: int = 100):
    """
    Create a custom split condition function based on a threshold for hexagon occurrences.

    This function returns a nested `split_condition` function that evaluates whether
    a hexagon should be split based on the number of occurrences compared to a specified
    threshold.

    Arguments:
    - threshold: int, the minimum number of occurrences required for a hexagon to be
                 considered for splitting. Default is 100.

    Returns:
    - A function `split_condition(current_res, hex_count, neighbors_stat)` that takes:
      - current_res: int, the current resolution of the hexagon (not used in this function).
      - hex_count: int, the count of occurrences in the current hexagon.
      - neighbors_stat: any, information/statistics about neighboring hexagons
                        (not used in this function).
      The `split_condition` function returns `True` if `hex_count` exceeds the threshold,
      indicating the hexagon should be split; otherwise, it returns `False`.
    """
    def split_condition(current_res, hex_count, neighbors_stat):
        """
        Decide if a hexagon should split based on its occurrence count.

        Arguments:
        - current_res: int, the current resolution of the hexagon (not used in this function).
        - hex_count: int, the count of occurrences in the current hexagon.
        - neighbors_stat: any, information/statistics about neighboring hexagons
                          (not used in this function).

        Returns:
        - True if `hex_count` exceeds the specified `threshold`, indicating the hexagon
          should be split; otherwise, False.
        """
        return hex_count > threshold

    return split_condition


def complex_split_condition(threshold=100, std_ratio=2):
    """
    Create a split condition function that decides whether to split a hexagon based on:
    1. The normalized occurrence count (using the coefficient of variation) exceeding a threshold.
    2. Significant normalized variance in the number of occurrences between the hexagon and its neighbors.

    Arguments:
    - threshold_ratio: float, the minimum coefficient of variation for the hexagon's count to consider splitting.
    - variance_ratio: float, the minimum normalized variance between the hexagon and its neighbors for significant change.

    Returns:
    - A function `split_condition(current_res, hex_count, neighbors_stat)` that returns True if the
      hexagon meets the split criteria; otherwise, False.
    """
    def split_condition(current_res, hex_count, neighbors_stat):
        """
        Determine if a hexagon should be split based on normalized variance and threshold ratio.

        Arguments:
        - current_res: int, the current resolution of the hexagon.
        - hex_count: int, the number of occurrences in the current hexagon.
        - neighbors_stat: dict, where keys are neighbor hex IDs and values are their occurrence counts.

        Returns:
        - Boolean: True if the hexagon meets the split conditions; otherwise, False.
        """
        if not neighbors_stat:
            return False

        neighbor_counts = np.array(list(neighbors_stat.values()))
        mean_neighbors = np.mean(neighbor_counts)
        if mean_neighbors == 0:
            return False

        if hex_count <= threshold:
            return False

        normalized_std = np.std(neighbor_counts) / mean_neighbors

        return normalized_std > std_ratio

    return split_condition


def skewness_stopping_condition(threshold: float = 0.2):
    """
    Create a stopping condition function based on the skewness of a distribution.

    This function returns a nested `stopping_condition` function that evaluates whether
    the skewness of the distribution of hexagon occurrences meets a specified threshold.
    The stopping condition is used to decide if the iterative process should halt.

    Arguments:
    - threshold: float, the skewness threshold for stopping. If the skewness of the
                 distribution of occurrences is less than this threshold, the condition
                 returns `True`, indicating that the process should stop. Default is 0.2.

    Returns:
    - A function `stopping_condition(hexagons)` that takes:
      - hexagons: dict[str, int], a dictionary where keys are hexagon IDs and values are
                  the count of occurrences in each hexagon.
      The function returns `True` if the skewness of the values in `hexagons` is below
      the specified threshold, indicating that the stopping condition is met; otherwise,
      it returns `False`.
    """
    def stopping_condition(hexagons: dict):
        """
        Evaluate the skewness of the hexagon occurrence counts to determine if the process should stop.

        Arguments:
        - hexagons: dict[str, int], a dictionary where keys are hexagon IDs and values are
                    the count of occurrences in each hexagon.

        Returns:
        - True if the skewness is less than the specified threshold, indicating that the
          condition for stopping is met; otherwise, False.
        """
        data = np.array(list(hexagons.values()))
        n = len(data)
        if n < 3:
            raise ValueError("Skewness calculation requires at least 3 data points.")

        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std_dev) ** 3)
        print(skewness)
        return skewness < threshold

    return stopping_condition


def mixed_resolution(
    split_condition_fn: Callable,
    stopping_condition_fn: Callable,
    dataset: str = 'geolife',
    min_resolution: int = 7,
    max_resolution: int = 9,
    input_dir: str = 'data/mixed_res',
    output_dir: str = 'data/mixed_res',
    max_iterations: int = 5,
):
    """
    Perform iterative hexagon refinement based on split and stopping conditions.

    This function reads preprocessed data including hexagon counts, unique hexagon sets,
    and neighbor relationships, and iteratively refines the hexagon set by splitting
    hexagons that meet the given `split_condition_fn`. The process halts when the
    `stopping_condition_fn` is satisfied or the maximum number of iterations is reached.

    Arguments:
    - split_condition_fn: Callable, a function that takes the current resolution,
                          hexagon count, and neighbor statistics and returns a boolean
                          indicating whether the hexagon should be split.
    - stopping_condition_fn: Callable, a function that takes a dictionary of hexagons
                             and their counts and returns a boolean indicating whether
                             the stopping condition is met.
    - dataset: str, the name of the dataset to process (e.g., 'geolife').
    - min_resolution: int, the initial resolution of hexagons to start processing from.
    - max_resolution: int, the maximum resolution allowed for splitting hexagons.
    - input_dir: str, the directory path to load the preprocessed input files.
    - output_dir: str, the directory path to save any output if necessary.
    - max_iterations: int, the maximum number of iterations for the process.

    Process:
    1. Load preprocessed neighbor relationships, hexagon counts, and unique hexagon sets
       from pickle files.
    2. Initialize the hexagon set from the `min_resolution` level.
    3. Iterate up to `max_iterations` times, splitting hexagons that meet the `split_condition_fn`.
    4. Check the `stopping_condition_fn` at each iteration to potentially stop the process early.
    5. Split marked hexagons into their children at the next higher resolution.
    6. Print status messages and information about each iteration.

    Outputs:
    - Prints the number of hexagons processed at each iteration.
    - Stops the process when the stopping condition is met or no hexagons meet the split condition.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{input_dir}/neighbors_{dataset}.pkl', 'rb') as f:
        neighbors = pickle.load(f)

    with open(f'{input_dir}/hex_count_{dataset}.pkl', 'rb') as f:
        hex_counts = pickle.load(f)

    with open(f'{input_dir}/hexes_{dataset}.pkl', 'rb') as f:
        hexes = pickle.load(f)

    hexagon_set = hexes[min_resolution]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}: Dataset has {len(hexagon_set)} hexagons")
        if stopping_condition_fn({hexagon: hex_counts[hexagon] for hexagon in hexagon_set}):
            print(f"Stopping condition invoked at iteration {iteration}")
            break

        marked_for_split = set()

        for hex_id in hexagon_set:
            current_res = h3.get_resolution(hex_id)
            if current_res == max_resolution:
                continue
            neighbors_stat = {neighbor: hex_counts[neighbor] for neighbor in neighbors[hex_id]}
            hex_count = hex_counts[hex_id]

            if split_condition_fn(current_res, hex_count, neighbors_stat):
                marked_for_split.add(hex_id)

        if len(marked_for_split) == 0:
            print(f"None of hexagons met split condition at iteration {iteration}")
            break
        else:
            print(f"{len(marked_for_split)} of hexagons met split condition at iteration {iteration}")

        new_hexagon_set = set()
        for hexagon in hexagon_set:
            if hexagon in marked_for_split:
                new_hexagon_set.update(h3.cell_to_children(hexagon))
            else:
                new_hexagon_set.add(hexagon)
        hexagon_set = new_hexagon_set

    with open(f'{output_dir}/final_hexes_{dataset}.pkl', 'wb') as f:
        pickle.dump(hexagon_set, f)
    print("Saved final hexagons set pickle")

    visualize({hexagon: h3.get_resolution(hexagon) for hexagon in hexagon_set}, "mixed-res-heatmap.pdf", categorical=True, cmap=ListedColormap(["#66CDAA", "#9370DB", "#86DAE3"]), legend_kwds={"loc": "upper right", "title":"Resolution", "markerscale":2.5},)
    visualize({hexagon: hex_counts[hexagon] for hexagon in hexes[min_resolution]}, "mixed-res-map.pdf", bins=[0,500,10000,np.inf], cmap=ListedColormap(["#66CDAA", "#9370DB", "#86DAE3"]), categorical=True, k=10, legend_kwds={"loc": "upper right", "title":"Movement Density", "markerscale":2.5,},)
    print("Heatmap saved")


def apply_processing(
    dataset: str = 'geolife',
    min_resolution: int = 7,
    max_resolution: int = 9,
    output_dir: str = 'data/mixed_res',
    date_column: str = "time"
    ):

    with open(f'{output_dir}/{dataset}.pkl', 'rb') as f:
        hexes = pickle.load(f)

    with open(f'{output_dir}/final_hexes_{dataset}.pkl', 'rb') as f:
        hexagon_set = pickle.load(f)


    boundary = {
        'geolife':{
            "Min_lat":39.50000000,
            "Max_lat":40.50000000,
            "Min_lon":116.00000000,
            "Max_lon":117.00000000
        },
        'rome':{
            "Min_lat":41.793710,
            "Max_lat":41.991390,
            "Min_lon":12.372598,
            "Max_lon":12.622537
        },
        'porto': {
            "Min_lat": -1000,
            "Max_lat": 1000,
            "Min_lon": -1000,
            "Max_lon": 1000
        }
    }

    def gps_to_mixed_h3(gps_points: list, boundary: dict):
        result = []
        for lon, lat in gps_points:
            if boundary["Min_lon"] <= lon and boundary["Min_lat"] <= lat and \
                boundary["Max_lon"] >= lon and boundary["Max_lat"] >= lat:
                for resolution in range(min_resolution, max_resolution+1):
                    cell = h3.latlng_to_cell(lat, lon, resolution)
                    if cell in hexagon_set:
                        if len(result) == 0 or cell != result[-1]:
                            result.append(cell)
                        break
        return result

    hexes['points'] = hexes['points'].apply(gps_to_mixed_h3, args=(boundary[dataset],))

    output_dir = Path(output_dir)
    (output_dir / dataset).mkdir(parents=True, exist_ok=True)

    vocab = ["EOT"] + list(hexagon_set)
    vocab_file_path = output_dir / dataset / 'vocab.txt'
    with vocab_file_path.open('w', encoding='utf-8') as vocab_file:
        vocab_file.write("\n".join(vocab) + "\n")

    mapping = {k: v for v, k in enumerate(vocab)}
    mapping_file_path = output_dir / dataset / 'mapping.json'
    with mapping_file_path.open('w', encoding='utf-8') as mapping_file:
        json.dump(mapping, mapping_file, ensure_ascii=False)

    hexes['points'] = hexes['points'].apply(lambda x: [str(mapping[j]) for j in x])
    df_mapped = hexes.sort_values(by=[date_column])["points"].to_list()
    data_file_path = output_dir / dataset / 'data.txt'
    with data_file_path.open('w', encoding='utf-8') as data_file:
        for item in df_mapped:
            data_file.write(' '.join(item) + f" {mapping['EOT']}\n")


def main() -> None:
    """
    Main function to handle argument parsing and execute data preprocessing.
    """
    parser = argparse.ArgumentParser(description='Trajectory Prediction Learning')
    parser.add_argument('dataset', type=str, help='Dataset')
    args = parser.parse_args()

    preprocess_resolution(dataset=args.dataset)
    mixed_resolution(
        split_condition_fn=threshold_split_condition(150),
        stopping_condition_fn=skewness_stopping_condition(0.2),
        dataset=args.dataset
        )
    apply_processing(dataset=args.dataset)


if __name__ == '__main__':
    main()
