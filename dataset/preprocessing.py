"""
preprocessing.py

This module provides preprocessing functions for earthquake signal data and ambient noise data.
It includes metadata extraction from header files, preprocessing of raw waveform files into
structured CSV files for earthquake events, and normalization of noise data.

Functions:
- extract_metadata(file_path): Extracts depth and magnitude metadata from a file.
- preprocess_eq_all(input_dir, output_dir): Converts and preprocesses earthquake waveform data into CSV format.
- preprocess_noise_all(source_folder, destination_folder): Normalizes and saves noise data from CSV files.
"""

import os
import numpy as np
import pandas as pd
import re
from obspy import read


def extract_metadata(file_path):
    """
    Extracts the earthquake metadata (depth and magnitude) from the given file.

    Args:
        file_path (str): Path to the file containing metadata in plain text format.

    Returns:
        tuple: A tuple (depth, magnitude) where both values are floats or None if not found.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    depth = magnitude = None
    for line in lines:
        if "Depth" in line:
            try:
                depth = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)[0])
            except:
                pass
        if "Mag" in line:
            try:
                magnitude = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)[0])
            except:
                pass
    return depth, magnitude


def preprocess_eq_all(input_dir, output_dir):
    """
    Preprocesses all earthquake waveform data from a directory.

    This function searches for triplets of waveform files (EW, NS, UD) in subdirectories of `input_dir`,
    applies scaling and calibration, removes mean offset, extracts metadata (depth and magnitude),
    and saves the result as a CSV file in `output_dir`.

    Args:
        input_dir (str): Directory containing raw waveform files.
        output_dir (str): Directory where the processed CSV files will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, _ in os.walk(input_dir):
        for sub_dir in dirs:
            second_level_path = os.path.join(root, sub_dir)
            for third_root, _, files in os.walk(second_level_path):
                ew_file = ns_file = ud_file = None
                for file in files:
                    if file.endswith(".EW"):
                        ew_file = os.path.join(third_root, file)
                    elif file.endswith(".NS"):
                        ns_file = os.path.join(third_root, file)
                    elif file.endswith(".UD"):
                        ud_file = os.path.join(third_root, file)

                if ew_file and ns_file and ud_file:
                    try:
                        ew, ns, ud = read(ew_file)[0], read(ns_file)[0], read(ud_file)[0]
                        sr = ew.stats.sampling_rate
                        timestamps = np.arange(0, len(ew.data) / sr, 1 / sr)
                        scale = 0.00101972 * 100
                        x = ew.data * scale * ew.stats.calib
                        y = ns.data * scale * ns.stats.calib
                        z = ud.data * scale * ud.stats.calib
                        x -= np.mean(x)
                        y -= np.mean(y)
                        z -= np.mean(z)
                        depth, mag = extract_metadata(ew_file)
                        df = pd.DataFrame({
                            "timestamp": timestamps,
                            "x": x, "y": y, "z": z,
                            "depth_km": depth, "magnitude": mag
                        })
                        filename = os.path.splitext(os.path.basename(ew_file))[0] + ".csv"
                        df.to_csv(os.path.join(output_dir, filename), index=False)
                    except Exception as e:
                        print(f"Error processing {ew_file}: {e}")


def preprocess_noise_all(source_folder, destination_folder):
    """
    Normalizes noise data from CSV files in the source folder and saves the processed files.

    The function standardizes acceleration data (x, y, z) by subtracting the mean and dividing by gravity (9.80665 m/s²).

    Args:
        source_folder (str): Directory containing input noise CSV files.
        destination_folder (str): Directory where normalized CSV files will be saved.

    Returns:
        None
    """
    os.makedirs(destination_folder, exist_ok=True)

    for subfolder_name in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder_name)

        if os.path.isdir(subfolder_path):
            output_subfolder = os.path.join(destination_folder, subfolder_name)
            os.makedirs(output_subfolder, exist_ok=True)

            for filename in os.listdir(subfolder_path):
                if filename.endswith(".csv"):
                    input_path = os.path.join(subfolder_path, filename)
                    output_path = os.path.join(output_subfolder, filename)

                    try:
                        df = pd.read_csv(input_path)
                        if all(c in df.columns for c in ['x', 'y', 'z']):
                            for axis in ['x', 'y', 'z']:
                                df[axis] = (df[axis] - np.mean(df[axis])) / 9.80665
                            df.to_csv(output_path, index=False)
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")
