# TrajLearn: A Novel Model for Trajectory Prediction

## Overview

Trajectory prediction is a critical task with wide-ranging applications in autonomous vehicles, robotics, human motion analysis, and more. The goal is to estimate the future path of an entity based on its current state and past movements. Current deep learning models often face challenges in accurately capturing complex spatial dependencies due to the dynamic nature of environments.

*TrajLearn* is our proposed solution — a transformer-based model designed to predict future trajectories using higher-order mobility flow representations (hexagonal grids). It integrates a beam search variant to enhance spatial continuity, leading to improved accuracy over existing methods. TrajLearn marks a substantial advancement in trajectory prediction models by combining generative approaches with higher-order spatial reasoning.

## Repository Structure

This project consists of several core components required for training, evaluating, and testing the TrajLearn model:

- `TrajectoryBatchDataset.py`: Handles dataset loading, processing, and batching.
- `config.yaml`: Configuration file for model training and evaluation, containing parameters such as batch size, learning rates, and - dataset-specific configurations.
- `environment.yml`: Specifies dependencies and environment settings required to run the project.
- `main.py`: The main script for training and testing the TrajLearn model.
- `model.py`: Contains the architecture and design of the TrajLearn model.
- `test.py`: Script to test and evaluate model performance.
- `trainer.py`: Manages the training process, including checkpoints and logging.
- `download_data.sh`: Script to download required datasets.
- `prepare.py`: Script to transform datasets into the required format.

## Configuration (`config.yaml`)

The `config.yaml` file contains various parameters for model training, validation, and testing. Below is a sample configuration for the `rome7` dataset:

```yaml
rome7-12-8-512:
  test_ratio: 0.2
  validation_ratio: 0.1
  delimiter: " "
  min_input_length: 5
  max_input_length: 20
  test_input_length: 10
  test_prediction_length: 5
  batch_size: 128
  device: cuda
  max_epochs: 4
  ...
  model_checkpoint_directory: /local/traj-pred/models/
```

## Getting Started

### Prerequisites

- This implementation requires **Python version >= 3.8**.
- Ensure you have a compatible Python environment. Refer to `environment.yml` for the required packages.

### Setting up the Dataset

1. **Download the Datasets**:
   - First, make the dataset downloader script executable:
     ```bash
     chmod +x ./download_data.sh
     ```
   - Run the script to download the datasets. By default, it downloads the **Geolife** dataset. You can specify the dataset by passing an argument (`geolife`, `porto`, or `rome`). For example:
     ```bash
     ./download_data.sh porto
     ```
     If no argument is provided, the **Geolife** dataset is downloaded by default.

2. **Prepare the Datasets**:
   - After downloading, run the following command to prepare and transform the datasets:
     ```bash
     python3 prepare.py --input_dir <input_directory> --output_dir <output_directory> --datasets <geolife|porto|rome>
     ```
   - You can specify the `input_dir`, `output_dir`, and `datasets` to be processed:
     - **`--input_dir`**: Directory where the raw datasets are stored. Defaults to `./data`.
     - **`--output_dir`**: Directory where the transformed datasets will be saved. Defaults to `./data`.
     - **`--datasets`**: Select which datasets to process (`geolife`, `porto`, `rome`). Multiple datasets can be processed by specifying them in a space-separated list. For example:
     ```bash
     python3 prepare.py --datasets geolife porto
     ```


### Training the Model

To train the TrajLearn model, use the following command:

```bash
python3 main.py config.yaml
```

### Testing the Model

To test the trained model, run the following command:

```bash
python3 main.py config.yaml --test
```

<img src="/amir-ni/Trajectory-prediction/raw/main/img/overview.png" alt="drawing" width="75%">
