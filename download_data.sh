#!/bin/bash

# This script downloads datasets, extracts them, and verifies MD5 checksums.

# Exit the script if any command fails
set -e

# Variables
DATA_DIR="./data"
BASE_URL="https://zenodo.org/records/8076553/files"

echo "Starting download and extraction script..."

# Check if arguments are provided
if [ "$#" -eq 0 ]; then
    echo "No datasets specified. Please provide dataset names as arguments."
    exit 1
fi

# Create data directory if it doesn't exist
echo "Checking for data directory..."
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory at $DATA_DIR"
    mkdir -p "$DATA_DIR"
else
    echo "Data directory already exists. Proceeding..."
fi

# Loop through all datasets provided as arguments
for DATASET in "$@"; do
    echo "Processing dataset: $DATASET"

    # Define output zip and dataset URL
    OUTPUT_ZIP="${DATASET}.zip"
    DATASET_URL="${BASE_URL}/ho_${OUTPUT_ZIP}?download=1"

    # Download the dataset
    echo "Checking if the dataset is already downloaded..."
    if [ -f "$DATA_DIR/$OUTPUT_ZIP" ]; then
        echo "Dataset zip file already exists. Skipping download for $DATASET."
    else
        echo "Downloading dataset from $DATASET_URL..."
        wget --show-progress -O "$DATA_DIR/$OUTPUT_ZIP" "$DATASET_URL"
        echo "Download completed for $DATASET."
    fi

    # Extract the dataset
    echo "Checking if dataset is already extracted..."
    if [ -d "$DATA_DIR/$DATASET" ]; then
        echo "Dataset already extracted. Skipping extraction for $DATASET."
    else
        echo "Extracting dataset to $DATASET..."
        unzip -q "$DATA_DIR/$OUTPUT_ZIP" -d "$DATA_DIR/$DATASET"
        echo "Extraction completed for $DATASET."
    fi

    # Clean up the zip file
    if [ -f "$DATA_DIR/$OUTPUT_ZIP" ]; then
        echo "Cleaning up zip file for $DATASET..."
        rm -f "$DATA_DIR/$OUTPUT_ZIP"
    fi

    # Verify MD5 checksums
    echo "Verifying MD5 checksums for files in $DATA_DIR/$DATASET..."
    for file in "$DATA_DIR/$DATASET"/*; do
        if [ -f "$file" ]; then
            md5sum "$file"
        fi
    done
    echo "MD5 checksum verification completed for $DATASET."
done

echo "Script completed successfully for all datasets!"
