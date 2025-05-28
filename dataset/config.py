"""
config.py

This module defines configuration constants for data processing in an earthquake 
detection and classification project. It includes paths for raw and processed data, 
feature extraction outputs, and general processing parameters.

Attributes:
    EQ_RAW_FOLDER (str): Path to the raw earthquake acceleration data (0.1g threshold) from 2020.
    EQ_PROCESSED_FOLDER (str): Path to the processed earthquake data.

    NOISE_RAW_FOLDER (str): Path to the raw non-earthquake (noise) data (e.g., walking).
    NOISE_PROCESSED_FOLDER (str): Path to the processed non-earthquake data.

    FEATURE_EQ_SLIDING (str): Path to the extracted features of earthquake data using sliding window method.
    FEATURE_NOISE_SINGLE (str): Path to the extracted features of non-earthquake data using single-row method.

    FINAL_DATASET_PATH (str): Path to the final merged dataset used for model training or evaluation.

    SAMPLING_RATE (int): Sampling rate of the time-series data in Hz.
    WINDOW_DURATION (int): Duration of each sliding window in seconds.
    OVERLAP (float): Overlap ratio between consecutive windows (0 to 1).

Usage:
    Import this module to access standardized paths and parameters across different scripts in the project.
"""


EQ_RAW_FOLDER = "/content/gdrive/MyDrive/De_tai_Tot_Nghiep/DATASET/K-NET/EQ/0.1g/2020"
EQ_PROCESSED_FOLDER = "/content/gdrive/MyDrive/KLTN_Nguyen_Ngoc_Thai/DATASET/Processing_DATA/0.1g"

NOISE_RAW_FOLDER = "/content/gdrive/MyDrive/De_tai_Tot_Nghiep/DATASET/K-NET/NonEQ/knuEqAccData_Walking"
NOISE_PROCESSED_FOLDER = "/content/gdrive/MyDrive/De_tai_Tot_Nghiep/DATASET/Processing_DATA/Non_EQ/Walking"

FEATURE_EQ_SLIDING = "/content/gdrive/MyDrive/De_tai_Tot_Nghiep/DATASET/Feature_EQ/feature_EQ_sliding.csv"
FEATURE_NOISE_SINGLE = "/content/gdrive/MyDrive/De_tai_Tot_Nghiep/DATASET/Feature_NonEQ/feature_Noise_single.csv"

FINAL_DATASET_PATH = "/content/gdrive/MyDrive/De_tai_Tot_Nghiep/DATASET/final_dataset.csv"

SAMPLING_RATE = 100
WINDOW_DURATION = 10
OVERLAP = 0.5
