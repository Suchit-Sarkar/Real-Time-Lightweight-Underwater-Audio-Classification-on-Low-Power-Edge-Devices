import os
import pandas as pd
import numpy as np
import torchaudio
import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm
import scipy.signal as signal

# List of directories and corresponding numeric values
# Species mapping
species_mapping = {
    "AtlanticSpottedDolphin":0,
    "BottlenoseDolphin":1,
    "Boutu_AmazonRiverDolphin":2,
    "ClymeneDolphin":3,
    "Commerson'sDolphin":4,
    "CommonDolphin": 5,
    "DuskyDolphin":6,
    "Fraser'sDolphin":7,
    "Grampus_Risso'sDolphin":8,
    "Heaviside'sDolphin":9,
    "IrawaddyDolphin":10,
    "noise":11,
    "LongBeaked(Pacific)CommonDolphin":12,
    "PantropicalSpottedDolphin": 13,
    "Rough_ToothedDolphin":14,
    "SpinnerDolphin": 15,
    "StripedDolphin": 16,
    "TucuxiDolphin":17,
    "White_beakedDolphin":18,
    "White_sidedDolphin": 19,
}


# Base directory containing the species folders
base_directory = "E:/Project_Experiments/only_dolphin_experiment/separated_augmented/test_data"

output_file = os.path.join(base_directory, "test_metadata.xlsx")

# Normalization factors for each feature
norm_factors = [
    1000.0,     # spectral_centroid (~880)
    1.0,        # rms_energy (~0.09)
    0.01,       # zero_crossing_rate (~0.005)
    1000.0,     # spectral_rolloff (~1087)
    1.0,        # band_energy_ratio (~0.08)
    500.0,      # MFCC1 (~-380)
    200.0,      # MFCC2 (~107)
    50.0,       # MFCC3 (~31)
    10.0,       # MFCC4 (~6.3)
    5.0,        # MFCC5 (~2.6)
    20.0,       # MFCC6 (~19)
    2.0,        # MFCC7 (~1.0)
    20.0,       # MFCC8 (~15)
    10.0,       # MFCC9 (~6.1)
    10.0,       # MFCC10 (~9.7)
    10.0,       # MFCC11 (~5.2)
    5.0,        # MFCC12 (~2.0)
    5.0         # MFCC13 (~4.2)
]

def _extract_additional_features(signal, sr):
    # Ensure the signal is mono
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=0)
    
    # Convert to 32-bit float if not already
    signal = signal.astype(np.float32)
    
    # Compute features
    spectral_centroid = np.float32(librosa.feature.spectral_centroid(y=signal, sr=sr).mean())
    rms_energy = np.float32(librosa.feature.rms(y=signal).mean())
    zero_crossing_rate = np.float32(librosa.feature.zero_crossing_rate(y=signal).mean())
    spectral_rolloff = np.float32(librosa.feature.spectral_rolloff(y=signal, sr=sr).mean())
    
    # Extract MFCCs as 32-bit floats
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).mean(axis=1).astype(np.float32)
    
    # Band energy ratio (example: ratio of energy in 4-8 kHz band to total energy)
    stft = np.abs(librosa.stft(signal))
    freq_range = librosa.fft_frequencies(sr=sr)
    
    # Adjust frequency bands based on sample rate
    lower_freq = min(1000, sr // 4)  # Lower bound or quarter of sample rate, whichever is smaller
    upper_freq = min(24000, sr // 2 - 1000)  # Upper bound or slightly below Nyquist, whichever is smaller
    
    band_mask = (freq_range >= lower_freq) & (freq_range <= upper_freq)
    band_energy_ratio = np.float32(np.sum(stft[band_mask, :]) / np.sum(stft) if np.sum(stft) > 0 else 0)
    
    # Combine all features into a single array (all as float32)
    additional_features = np.concatenate([
        [spectral_centroid, rms_energy, zero_crossing_rate, spectral_rolloff, band_energy_ratio],
        mfccs
    ]).astype(np.float32)
    
    return additional_features

def _cut_if_necessary(signal, num_samples):
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal

def _right_pad_if_necessary(signal, num_samples):
    if signal.shape[1] < num_samples:
        num_missing_samples = num_samples - signal.shape[1]
        last_dim_padding = (0, num_missing_samples)
        signal = F.pad(signal, last_dim_padding)
    return signal

def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def process_audio_file(file_path):
    try:
        # Load audio file with original sample rate
        signal, sr = torchaudio.load(file_path)
        
        # Calculate the number of samples for 5 seconds of audio at the original sample rate
        num_samples = 480000  # 5 seconds of audio at original sample rate
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=96000)
        signal = resampler(signal)
        # Ensure the audio is in float32 format
        signal = signal.to(torch.float32)
        
        # Preprocess audio
        signal = _mix_down_if_necessary(signal)
        signal = _cut_if_necessary(signal, num_samples)
        signal = _right_pad_if_necessary(signal, num_samples)
        
        # Extract features
        #additional_features = _extract_additional_features(signal.cpu().numpy().squeeze(), sr)
        
        signal = signal.cpu().numpy()
        signal = _process_audio(signal, sr)
        signal = torch.tensor(signal.copy(), dtype=torch.float32)
        additional_features = _extract_additional_features(signal.cpu().numpy().squeeze(), sr)
        
        feature_dict = {
            'sample_rate': sr,  # Store the original sample rate
            'spectral_centroid': additional_features[0],
            'rms_energy': additional_features[1],
            'zero_crossing_rate': additional_features[2],
            'spectral_rolloff': additional_features[3],
            'band_energy_ratio': additional_features[4]
        }
        
        # Add MFCCs to the dictionary
        for i, mfcc in enumerate(additional_features[5:]):
            feature_dict[f'mfcc_{i+1}'] = mfcc
            
        return feature_dict
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

#Bandpass filter
def _bandpass_filter(audio, sr, lowcut=1000, highcut=24000, order=4):
        try:
            nyquist = 0.5 * sr
            low = lowcut / nyquist
            high = highcut / nyquist
            # Ensure frequencies are between 0 and 1
            low = max(0.001, min(0.99, low))
            high = max(0.001, min(0.99, high))
        
            # Make sure low is less than high
            if low >= high:
                low, high = high * 0.5, high
                
            b, a = signal.butter(order, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            return filtered_audio
        except Exception as e:
            print(f"Error applying bandpass filter: {str(e)}")
            raise
    

def _process_audio(signal, sr):
        # Apply bandpass filter
    filtered_audio = _bandpass_filter(signal, sr, lowcut=1000, highcut=24000)
    return filtered_audio
# Data collection
data = []
print("Processing audio files and extracting features...")

for species_name, numeric_value in species_mapping.items():
    species_dir = os.path.join(base_directory, species_name)
    
    if os.path.exists(species_dir):
        files = [f for f in os.listdir(species_dir) if f.endswith(".wav")]
        
        for file_name in tqdm(files, desc=f"Processing {species_name}"):
            file_path = os.path.join(species_dir, file_name)
            features = process_audio_file(file_path)
            
            if features:
                row_data = [file_name, species_name, numeric_value, features['sample_rate']]
                
                # Add features to row data (excluding sample_rate )
                # Apply normalization factors to each feature
                feature_index = 0  # Track which normalization factor to use
                for feature_name, feature_value in features.items():
                    if feature_name != 'sample_rate':  
                        # Apply normalization to the feature
                        normalized_value = feature_value / norm_factors[feature_index]
                        row_data.append(normalized_value)
                        feature_index += 1
                
                data.append(row_data)
    else:
        print(f"Directory not found: {species_dir}")

# Create column names
column_names = ["file_name", "species_name", "numeric_value", "sample_rate", 
                "spectral_centroid", "rms_energy", "zero_crossing_rate", 
                "spectral_rolloff", "band_energy_ratio"]

# Add MFCC column names
for i in range(1, 14):  # 13 MFCCs
    column_names.append(f"mfcc_{i}")

# Create a DataFrame
df = pd.DataFrame(data, columns=column_names)


pd.set_option('display.precision', 8)  # Use full option name

# Save to Excel with float32 precision
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, float_format='%.8f')

print(f"Excel file generated with features: {output_file}")
print(f"Total files processed: {len(data)}")
print(f"All features are stored as 32-bit float values")
print(f"Features have been normalized using the provided normalization factors")