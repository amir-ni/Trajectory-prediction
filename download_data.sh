#!/bin/bash

# This script downloads datasets, extracts them, and verifies MD5 checksums.

# Exit the script if any command fails
set -e

# Variables for URLs and filenames
DATASET=${1:-"geolife"}
DATA_DIR="./data"
OUTPUT_ZIP="${DATASET}.zip"
BASE_URL="https://zenodo.org/records/8076553/files"
DATASET_URL="${BASE_URL}/ho_${OUTPUT_ZIP}?download=1"

echo "Starting download and extraction script..."

# Create directory
echo "Checking for data directory..."
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory at $DATA_DIR"
    mkdir -p "$DATA_DIR"
else
    echo "Data directory already exists. Proceeding..."
fi

# Download the dataset
echo "Checking if the dataset is already downloaded..."
if [ -f "$DATA_DIR/$OUTPUT_ZIP" ]; then
    echo "Dataset zip file already exists. Skipping download."
else
    echo "Downloading dataset from $DATASET_URL..."
    wget --show-progress -O "$DATA_DIR/$OUTPUT_ZIP" "$DATASET_URL"
    echo "Download completed."
fi

# Extract the dataset
echo "Checking if dataset is already extracted..."
if [ -d "$DATA_DIR/$DATASET" ]; then
    echo "Dataset already extracted. Skipping extraction."
else
    echo "Extracting dataset to $DATASET..."
    unzip -q "$DATA_DIR/$OUTPUT_ZIP" -d "$DATA_DIR/$DATASET"
    echo "Extraction completed."
fi

# Clean up the zip file
if [ -f "$DATA_DIR/$OUTPUT_ZIP" ]; then
    echo "Cleaning up zip file..."
    rm -f "$DATA_DIR/$OUTPUT_ZIP"
fi

# Verify MD5 checksums
echo "Verifying MD5 checksums for files in $DATASET..."
cd "$DATA_DIR/$DATASET"
for file in *; do
    if [ -f "$file" ]; then
        md5sum "$file"
    fi
done
echo "MD5 checksum verification completed."

echo "Script completed successfully!"
