"""
features.py

This module provides functions for extracting time-domain and frequency-domain features 
from three-axis acceleration data (x, y, z) used in earthquake (EQ) and non-earthquake (noise) detection.

It supports:
- Feature extraction from single time-series windows.
- Sliding window segmentation for time-series data.
- Combining features from earthquake and noise data into a labeled dataset.
- A complete pipeline for preprocessing and feature extraction.

Functions:
    zero_crossing_rate(signal): Calculates the zero-crossing rate of a signal.
    interquartile_range(signal): Computes the interquartile range (IQR) of a signal.
    extract_features(df): Extracts statistical and spectral features from x, y, z columns of a DataFrame.
    extract_sliding_features_all(data_folder, output_path, sr, duration, overlap): 
        Extracts features using a sliding window approach across all .csv files in a directory.
    extract_features_all_single_row(data_folder, output_path): 
        Extracts features from entire time-series (single row per file) across all .csv files in a directory.
    build_final_dataset(eq_feature_path, noise_feature_path, output_path): 
        Merges earthquake and noise features into a single labeled dataset.
    execute_feature_pipeline(): 
        Executes the full pipeline from preprocessing to feature extraction and dataset creation.

Usage:
    Run this module directly to execute the complete feature extraction pipeline:
        $ python features.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from config import *
from dataset.preprocessing import preprocess_eq_all, preprocess_noise_all


def zero_crossing_rate(signal):
    """
    Calculate the number of times the signal crosses zero.

    Args:
        signal (np.ndarray): A 1D signal array.

    Returns:
        int: The number of zero crossings in the signal.
    """
    return np.sum(np.diff(np.sign(signal)) != 0)


def interquartile_range(signal):
    """
    Compute the interquartile range (IQR) of the signal.

    Args:
        signal (np.ndarray): A 1D signal array.

    Returns:
        float: The interquartile range (Q3 - Q1) of the signal.
    """
    return np.percentile(signal, 75) - np.percentile(signal, 25)


def extract_features(df):
    """
    Extract statistical and spectral features from x, y, z acceleration data.

    Args:
        df (pd.DataFrame): A DataFrame with columns 'x', 'y', 'z' representing acceleration data.

    Returns:
        dict: A dictionary of features including mean, std, min, max, skewness, kurtosis,
              peak-to-peak, IQR, dominant frequency, energy, and zero-crossing rate for each axis.
    """
    features = {}
    for axis in ['x', 'y', 'z']:
        data = df[axis].values
        features[f'Mean_{axis}'] = np.mean(data)
        features[f'Std_{axis}'] = np.std(data)
        features[f'Max_{axis}'] = np.max(data)
        features[f'Min_{axis}'] = np.min(data)
        features[f'Peak_to_peak_{axis}'] = np.max(data) - np.min(data)
        features[f'Skew_{axis}'] = skew(data) if np.std(data) > 1e-6 else 0
        features[f'Kurtosis_{axis}'] = kurtosis(data) if np.std(data) > 1e-6 else 0
        features[f'IQR_{axis}'] = interquartile_range(data)
        fft_values = np.abs(fft(data))[:len(data)//2]
        features[f'Dominant_freq_{axis}'] = np.argmax(fft_values)
        features[f'Energy_{axis}'] = np.sum(fft_values**2)
        features[f'ZC_{axis}'] = zero_crossing_rate(data)
    return features


def extract_sliding_features_all(data_folder, output_path, sr, duration, overlap):
    """
    Extract features using a sliding window approach from all CSV files in a directory.

    Args:
        data_folder (str): Path to the folder containing input CSV files.
        output_path (str): Path to save the output CSV with extracted features.
        sr (int): Sampling rate in Hz.
        duration (int): Duration of each sliding window in seconds.
        overlap (float): Overlap ratio between consecutive windows (0 to 1).

    Returns:
        None
    """
    samples = int(sr * duration)
    step = int(samples * (1 - overlap))
    features_list = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                for start in range(0, len(df) - samples + 1, step):
                    segment = df.iloc[start:start + samples]
                    features_list.append(extract_features(segment))
    pd.DataFrame(features_list).to_csv(output_path, index=False)


def extract_features_all_single_row(data_folder, output_path):
    """
    Extract features from the entire signal of each CSV file (one feature row per file).

    Args:
        data_folder (str): Path to the folder containing input CSV files.
        output_path (str): Path to save the output CSV file with features.

    Returns:
        None
    """
    all_features = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                all_features.append(extract_features(df))
    pd.DataFrame(all_features).to_csv(output_path, index=False)


def build_final_dataset(eq_feature_path, noise_feature_path, output_path):
    """
    Combine earthquake and noise feature files into a single labeled dataset.

    Args:
        eq_feature_path (str): Path to the CSV file with earthquake features.
        noise_feature_path (str): Path to the CSV file with noise features.
        output_path (str): Path to save the final combined dataset with labels.

    Returns:
        None
    """
    eq_df = pd.read_csv(eq_feature_path)
    eq_df["label"] = 1
    noise_df = pd.read_csv(noise_feature_path)
    noise_df["label"] = 0
    df = pd.concat([eq_df, noise_df], ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Final dataset saved: {output_path}")


def execute_feature_pipeline():
    """
    Execute the full feature extraction pipeline.

    This includes:
        - Preprocessing of earthquake and noise data.
        - Feature extraction using both sliding window and single-row methods.
        - Merging features into a final labeled dataset.

    Returns:
        None
    """
    print("Starting the full processing pipeline...")
    preprocess_eq_all(EQ_RAW_FOLDER, EQ_PROCESSED_FOLDER)
    preprocess_noise_all(NOISE_RAW_FOLDER, NOISE_PROCESSED_FOLDER)
    extract_sliding_features_all(EQ_PROCESSED_FOLDER, FEATURE_EQ_SLIDING, SAMPLING_RATE, WINDOW_DURATION, OVERLAP)
    extract_features_all_single_row(NOISE_PROCESSED_FOLDER, FEATURE_NOISE_SINGLE)
    build_final_dataset(FEATURE_EQ_SLIDING, FEATURE_NOISE_SINGLE, FINAL_DATASET_PATH)


if __name__ == "__main__":
    execute_feature_pipeline()
