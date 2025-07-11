import time
import os
import pandas as pd
import numpy as np
import torchaudio
import librosa
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from noisereduce import reduce_noise
# Import VGGish (make sure torchvggish is installed)
from torchvggish import vggish
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import scipy.signal as signal
import timeit

# Custom Dataset for 9 Species Experiment
class MarineMammalDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, target_sample_rate, num_samples, device, transformation):
        self.annotations = pd.read_excel(annotation_file)  # Read Excel metadata with precomputed features
        self.audio_dir = audio_dir
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transformation = transformation
        
        # Encode labels as integers
        self.label_encoder = LabelEncoder()
        self.annotations['species_name'] = self.label_encoder.fit_transform(self.annotations['species_name'])
        # Add "unknown" class
        self.label_encoder.classes_ = np.append(self.label_encoder.classes_, "unknown")
        
        # Extract feature column names - update to use actual feature columns
        self.feature_columns = ['spectral_centroid', 'rms_energy', 
                               'zero_crossing_rate', 'spectral_rolloff', 'band_energy_ratio', 
                               'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 
                               'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 
                               'mfcc_11', 'mfcc_12', 'mfcc_13']
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        signal, sr = torchaudio.load(audio_sample_path)

        # Send to GPU
        signal = signal.to(self.device)

        # Handle the audio signal processing
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        
        # Apply bandpass filter
        signal = signal.cpu().numpy()
        signal = self._process_audio(signal, self.annotations.iloc[index]['sample_rate'])
        signal = torch.tensor(signal.copy(), dtype=torch.float32)
        
        # Generate mel spectrogram
        mel_spec = self.transformation(signal)
        mel_spec = torch.log(mel_spec + 1e-6)  # Convert to log scale for VGGish

        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(96, 64), mode='bilinear', align_corners=False
        ).squeeze(0)
        mel_spec = mel_spec.to(self.device)
        
        # Ensure 3D shape
        mel_spec = mel_spec.unsqueeze(0) if mel_spec.ndim == 2 else mel_spec

        # Get pre-computed features from Excel file
        additional_features = self._get_precomputed_features(index)
        additional_features = torch.tensor(additional_features, dtype=torch.float32).to(self.device)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long).to(self.device)
        
        return mel_spec, additional_features, label
    
    def _get_precomputed_features(self, index):
    # Extract features one by one and ensure they're numeric
        features = []
        for col in self.feature_columns:
        # Convert each value to float, with 0 as fallback for non-numeric values
            try:
                val = float(self.annotations.iloc[index][col])
            except (ValueError, TypeError):
                val = 0.0
            features.append(val)
    
        return np.array(features, dtype=np.float32)
    def _clip_waveform(self, signal, clip_value=0.99):
        return torch.clamp(signal, -clip_value, clip_value)
    
    def _process_audio(self, signal, sr):
        # Apply bandpass filter
        filtered_audio = self._bandpass_filter(signal, sr, lowcut=3000, highcut=19000)
        return filtered_audio
    
    def _bandpass_filter(self, audio, sr, lowcut=3000, highcut=19000, order=4):
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
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        file_name = self.annotations.iloc[index]['file_name']  # Using column name instead of index
        path = os.path.join(self.audio_dir, file_name)
        return os.path.normpath(path)

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index]['species_name']  # Using column name instead of index



# Load the large model

def load_model_for_inference(model_path, device):
    # Define the same model architecture modifications
    def create_fc(num_inputs, num_layers, neuron_per_layer, num_outputs, dropout_rate):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(num_inputs, neuron_per_layer))
            layers.append(nn.LayerNorm(neuron_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate, inplace=False))
            num_inputs = neuron_per_layer
            neuron_per_layer = num_inputs // 2
        layers.append(nn.Linear(num_inputs, num_outputs))
        return nn.Sequential(*layers)

    # Load with weights_only=False since we saved numpy arrays
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Reconstruct the model
    model = vggish().to(device)
    model.fc = create_fc(12288 + checkpoint['num_additional_features'], 4, 512, 20, 0.2).to(device)
    
    # Define the forward method
    def new_forward(x, additional_features):
        with torch.set_grad_enabled(True):
            vggish_embeddings = model.features(x).view(x.size(0), -1)
            if additional_features is not None:
                combined_features = torch.cat([vggish_embeddings, additional_features], dim=1)
            else:
                combined_features = vggish_embeddings
            x = model.fc(combined_features)
        return x
    
    model.forward = new_forward
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = checkpoint['label_encoder_classes']

    return model, {
        'label_encoder': label_encoder,
        'confidence_threshold': checkpoint['confidence_threshold'],
        'feature_columns': checkpoint['feature_columns'],
        'sample_rate': checkpoint['sample_rate'],
        'num_samples': checkpoint['num_samples']
    }

# Usage
device = "cuda" if torch.cuda.is_available() else "cpu"

model, config = load_model_for_inference(
    "E:/Project_Experiments/only_dolphin_experiment/fine_tuned_marine_vggish_combined_with_noise.pth",
    device
)

# Define a smaller student model
class StudentModel(nn.Module):
    def __init__(self, num_additional_features, num_classes):
        super(StudentModel, self).__init__()
        # Simplified CNN for spectrogram processing
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # FC layers for combined features
        self.fc = nn.Sequential(
            nn.Linear(64 + num_additional_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes))
    
    def forward(self, x, additional_features):
        # Process spectrogram
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        
        # Combine with additional features
        if additional_features is not None:
            x = torch.cat([x, additional_features], dim=1)
        
        # Final classification
        x = self.fc(x)
        return x

class AudioTransformerStudent(nn.Module):
    def __init__(self, num_additional_features, num_classes):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 48x32
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)   # 24x16
        )
        
        # Replace complex attention with CBAM
        self.attention = nn.Sequential(
            CBAM(32),
            nn.Conv2d(32, 64, kernel_size=1)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optional: Process additional features
        self.additional_processor = nn.Sequential(
            nn.Linear(num_additional_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 256),  # 64 from audio + 32 from processed features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, additional_features):
        x = self.cnn(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Process additional features
        additional_features = self.additional_processor(additional_features)
        
        # Combine with additional features
        x = torch.cat([x, additional_features], dim=1)
        
        return self.classifier(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # Create spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention))
        return x * attention

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    
train_annotation_file = "E:/Project_Experiments/only_dolphin_experiment/train_metadata.xlsx"
train_audio_dir = "E:/Project_Experiments/only_dolphin_experiment/train"
test_annotation_file = "E:/Project_Experiments/only_dolphin_experiment/test_metadata.xlsx"
test_audio_dir = "E:/Project_Experiments/only_dolphin_experiment/test"
validate_annotation_file = "E:/Project_Experiments/only_dolphin_experiment/validate_metadata.xlsx"
validate_audio_dir = "E:/Project_Experiments/only_dolphin_experiment/validate"

# VGGish-specific configurations
    
device = "cuda" if torch.cuda.is_available() else "cpu"
sample_rate = 16000  # Required sample rate for VGGish
num_samples = sample_rate * 5  # 5 seconds of audio at 16 kHz
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,  # Smaller window size for VGGish
        hop_length=160,  # Matches VGGish expectation
        n_mels=64
    )

    # Create the dataset
train_dataset = MarineMammalDataset(train_annotation_file, train_audio_dir, sample_rate, num_samples, device, mel_spectrogram)
test_dataset = MarineMammalDataset(test_annotation_file, test_audio_dir, sample_rate, num_samples, device, mel_spectrogram)
validate_dataset = MarineMammalDataset(validate_annotation_file, validate_audio_dir, sample_rate, num_samples, device, mel_spectrogram)
    
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(validate_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
vgg_feat,add_feat,lab=train_dataset[5]
print("Vggish features",vgg_feat.shape,"\nAdditional features",add_feat, "\nNumber of additional features",len(add_feat))
num_additional_features=len(add_feat)
student = StudentModel(num_additional_features, 20).to(device)
student_with_attention = AudioTransformerStudent(num_additional_features,20).to(device)

# # Add this import at the top of your file
# # Add this right after your model definition (after model.fc creation)
# print("\n=== Model Architecture Summary ===")
# print(model)  # Basic architecture overview

# # For detailed summary (add after model definition)
# try:
#     from torchsummary import summary
#     print("\n=== Detailed Model Summary ===")
#     # Note: We need to provide input sizes for both mel_spec and additional_features
#     # Since summary can't handle multiple inputs directly, we'll show them separately
#     print("Mel Spectrogram path summary:")
#     summary(model.features, (1, 96, 64), device=device)
#     print("\nClassifier path summary:")
#     summary(model.fc, (12288 + 18,), device=device)
# except ImportError:
#     print("torchsummary not available, skipping detailed summary")

# Add this at the end of your script (before saving the model)
# print("\n=== Testing with Random Sample ===")
# # Generate random sample matching your input format
# random_mel_spec = torch.rand(1, 1, 96, 64).to(device)  # Batch of 1
# random_features = torch.rand(1, 18).to(device)
# print("randon feature shape = ",random_features.shape, "random_mel_spec shape= ",random_mel_spec.shape)

# # Test inference
# model.eval()
# with torch.no_grad():
#     output = model(random_mel_spec, random_features)
#     probs = torch.softmax(output, dim=1)
#     max_prob, pred_class = torch.max(probs, 1)
    
#     # Apply confidence threshold
#     if max_prob < config['confidence_threshold']:
#         pred_class = torch.tensor([len(config['label_encoder'].classes_) - 1], device=device)  # Unknown class
    
#     # Decode prediction using the loaded label encoder
#     pred_label = config['label_encoder'].inverse_transform(pred_class.cpu().numpy())
    
#     print(f"\nTest Inference Results:")
#     print(f"Input shape - Mel Spec: {random_mel_spec.shape}, Features: {random_features.shape}")
#     print(f"Raw output shape: {output.shape}")
#     print(f"Predicted class: {pred_class.item()} ({pred_label[0]})")
#     print(f"Confidence: {max_prob.item():.4f}")
#     if pred_class.item() == len(config['label_encoder'].classes_) - 1:
#         print("-> Classified as 'unknown' (low confidence)")

#############################################################################################################################################
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
##########################################################################################################################################
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
    lower_freq = min(3000, sr // 4)  # Lower bound or quarter of sample rate, whichever is smaller
    upper_freq = min(19000, sr // 2 - 1000)  # Upper bound or slightly below Nyquist, whichever is smaller
    
    band_mask = (freq_range >= lower_freq) & (freq_range <= upper_freq)
    band_energy_ratio = np.float32(np.sum(stft[band_mask, :]) / np.sum(stft) if np.sum(stft) > 0 else 0)
    
    # Combine all features into a single array (all as float32)
    additional_features = np.concatenate([
        [spectral_centroid, rms_energy, zero_crossing_rate, spectral_rolloff, band_energy_ratio],
        mfccs
    ]).astype(np.float32)
    #normalize additional features 
    feature_index = 0  # Track which normalization factor to use
    for feature_index in range(0,len(norm_factors)):
        
        additional_features[feature_index]=additional_features[feature_index]/norm_factors[feature_index]
        #print(additional_features[feature_index])
        feature_index += 1
                
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

def process_audio_file(file_path="E:/Project_Experiments/only_dolphin_experiment/separated_augmented/test_data/Grampus_Risso'sDolphin/5902900G.wav"):
    try:
        # Load audio file with original sample rate
        signal, sr = torchaudio.load(file_path)
        
        # Calculate the number of samples for 5 seconds of audio at the original sample rate
        num_samples = sr * 5  # 5 seconds of audio at original sample rate
        
        # Ensure the audio is in float32 format
        signal = signal.to(torch.float32)
        
        # Preprocess audio
        signal = _mix_down_if_necessary(signal)
        signal = _cut_if_necessary(signal, num_samples)
        signal = _right_pad_if_necessary(signal, num_samples)
        #print(type(signal), signal.shape)
        signal = signal.numpy()
        #print(type(signal))
        signal = _process_audio(signal, sr)
        # Extract features
        additional_features = _extract_additional_features(signal.squeeze(), sr)
        additional_features=torch.tensor(additional_features.copy(), dtype=torch.float32)


        signal = torch.tensor(signal.copy(), dtype=torch.float32)
        sample_rate = 16000  # Required sample rate for VGGish
        num_samples = sample_rate * 5  # 5 seconds of audio at 16 kHz
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,  # Smaller window size for VGGish
        hop_length=160,  # Matches VGGish expectation
        n_mels=64
        )
        
        # Generate mel spectrogram
        transformation=mel_spectrogram
        # Generate mel spectrogram
        mel_spec = transformation(signal)
        mel_spec = torch.log(mel_spec + 1e-6)  # Convert to log scale for VGGish

        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(96, 64), mode='bilinear', align_corners=False
        ).squeeze(0)
        mel_spec = mel_spec.to(device)
        
        # Ensure 3D shape
        mel_spec = mel_spec.unsqueeze(0) if mel_spec.ndim == 2 else mel_spec

        return mel_spec, additional_features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def _bandpass_filter(audio, sr, lowcut=3000, highcut=19000, order=4):
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
    filtered_audio = _bandpass_filter(signal, sr, lowcut=3000, highcut=19000)
    return filtered_audio


#############################################################################################################################################
audio_sample_path = "E:/Project_Experiments/only_dolphin_experiment/separated_augmented/test_data/Grampus_Risso'sDolphin/5902900G.wav"

def model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_in_bytes = total_params * 4  # Assuming 32-bit (4 bytes) per parameter
    size_in_mb = size_in_bytes / (1024 ** 2)  # Convert to MB
    return size_in_mb


sig, sr = torchaudio.load(audio_sample_path)       
        # Calculate the number of samples for 5 seconds of audio at the original sample rate
num_samples = sr * 5  # 5 seconds of audio at original sample rate
   # Ensure the audio is in float32 format

sig = sig.to(torch.float32)
def proc_audio(sig=sig):       
    sig = _mix_down_if_necessary(sig)
    sig = _cut_if_necessary(sig, num_samples)
    sig = _right_pad_if_necessary(sig, num_samples)
        #print(type(signal), signal.shape)
    sig = sig.numpy()
def filter_s(sig=sig,sr=sr):
        #print(type(signal))
    sig = _process_audio(sig, sr)

def add_feat(sig=sig,sr=sr):
    sig = sig.numpy()
    additional_features = _extract_additional_features(sig.squeeze(), sr)
    additional_features=torch.tensor(additional_features.copy(), dtype=torch.float32)
    sig = torch.tensor(sig.copy(), dtype=torch.float32)

def mel_s(sig=sig,sr=sr):
    #sig = torch.tensor(sig.copy(), dtype=torch.float32)
    sample_rate = 16000  # Required sample rate for VGGish
    num_samples = sample_rate * 5  # 5 seconds of audio at 16 kHz
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,  # Smaller window size for VGGish
        hop_length=160,  # Matches VGGish expectation
        n_mels=64
        )
        
        # Generate mel spectrogram
    transformation=mel_spectrogram
        # Generate mel spectrogram
    mel_spec = transformation(sig)
    mel_spec = torch.log(mel_spec + 1e-6)  # Convert to log scale for VGGish

    mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(96, 64), mode='bilinear', align_corners=False
        ).squeeze(0)
    mel_spec = mel_spec.to(device)
        
        # Ensure 3D shape
    mel_spec = mel_spec.unsqueeze(0) if mel_spec.ndim == 2 else mel_spec

preprocess_time=timeit.timeit(proc_audio, number=100) / 100
print(f"Preprocess time={preprocess_time}")
preprocess_time=timeit.timeit(filter_s, number=100) / 100
print(f"Filtering time={preprocess_time}")
preprocess_time=timeit.timeit(add_feat, number=100) / 100
print(f"Additional Feature extraction time={preprocess_time}")
preprocess_time=timeit.timeit(mel_s, number=100) / 100
print(f"mel-spectrogram calculation time={preprocess_time}")
    
# Benchmarking block
def benchmark_processing():
    process_audio_file(audio_sample_path)

# Run the benchmark using timeit
execution_time = timeit.timeit(benchmark_processing, number=100)
average_time = execution_time / 100

print(f"Average time to process audio file over 10 runs: {average_time:.4f} seconds")


#############################################################################################################################################

# def inference(model,audio_sample_path):
#     model.eval()
#     print("Checking the sample...")
#     print("Extracting mel_spec + Additional features...")
#     mel_spec,additional_features=process_audio_file(audio_sample_path)
#     mel_spec=mel_spec.unsqueeze(0)
#     additional_features=additional_features.unsqueeze(0)
#     with torch.no_grad():
#         output = model(mel_spec.to(device), additional_features.to(device))
#         probs = torch.softmax(output, dim=1)
#         max_prob, pred_class = torch.max(probs, 1)
    
#     # Apply confidence threshold
#         if max_prob < config['confidence_threshold']:
#             pred_class = torch.tensor([len(config['label_encoder'].classes_) - 1], device=device)  # Unknown class
#         return pred_class
# def inference_t(model=model,audio_sample_path=audio_sample_path):
#     model.eval()
#     #print("Checking the sample...")
#     #print("Extracting mel_spec + Additional features...")
#     mel_spec,additional_features=process_audio_file(audio_sample_path)
#     mel_spec=mel_spec.unsqueeze(0)
#     additional_features=additional_features.unsqueeze(0)
#     with torch.no_grad():
#         output = model(mel_spec.to(device), additional_features.to(device))
#         probs = torch.softmax(output, dim=1)
#         max_prob, pred_class = torch.max(probs, 1)
    
#     # Apply confidence threshold
#         if max_prob < config['confidence_threshold']:
#             pred_class = torch.tensor([len(config['label_encoder'].classes_) - 1], device=device)  # Unknown class
#         return pred_class
# def inference_s(model=student,audio_sample_path=audio_sample_path):
#     model.eval()
#     #print("Checking the sample...")
#     #print("Extracting mel_spec + Additional features...")
#     mel_spec,additional_features=process_audio_file(audio_sample_path)
#     mel_spec=mel_spec.unsqueeze(0)
#     additional_features=additional_features.unsqueeze(0)
#     with torch.no_grad():
#         output = model(mel_spec.to(device), additional_features.to(device))
#         probs = torch.softmax(output, dim=1)
#         max_prob, pred_class = torch.max(probs, 1)
    
#     # Apply confidence threshold
#         if max_prob < config['confidence_threshold']:
#             pred_class = torch.tensor([len(config['label_encoder'].classes_) - 1], device=device)  # Unknown class
#         return pred_class    
# # Measure execution time
# start_time = time.time()
# pred_class=inference(student,audio_sample_path)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.6f} seconds")
# # Decode prediction using the loaded label encoder
# pred_label = config['label_encoder'].inverse_transform(pred_class.cpu().numpy())
# print(f"\nTest Inference Results:")
# #print(f"Raw output shape: {output.shape}")
# print(f"Predicted class: {pred_class.item()} ({pred_label[0]})")
# #print(f"Confidence: {max_prob.item():.4f}")
# if pred_class.item() == len(config['label_encoder'].classes_) - 1:
#     print("-> Classified as 'unknown' (low confidence)")

# print("Checking teacher.......\n")
# execution_time_t = timeit.timeit(inference_t, number=100) / 100
# print("Checking student.......\n")
# execution_time_s = timeit.timeit(inference_s, number=100) / 100
# print(f"Size of teacher: {model_size(model)}, Size of student:{model_size(student)}")
# print(f"Inference time of teacher:{execution_time_t}, Inference time of student:{execution_time_s}")
feature_extraction_execution_time=timeit.timeit(process_audio_file, number=100) / 100
print(f"Feature Extraction time={feature_extraction_execution_time}")

print("Extracting mel_spec + Additional features...")
def extract_feature(audio_sample_path):
    mel_spec,additional_features=process_audio_file(audio_sample_path)
    mel_spec=mel_spec.unsqueeze(0)
    additional_features=additional_features.unsqueeze(0)
    return mel_spec,additional_features
mel_spec,additional_features=extract_feature(audio_sample_path)

model.eval()
student.eval()
student_with_attention.eval()
def inference(model):
    with torch.no_grad():
        output = model(mel_spec.to(device), additional_features.to(device))
        probs = torch.softmax(output, dim=1)
        max_prob, pred_class = torch.max(probs, 1)
    # Apply confidence threshold
        if max_prob < config['confidence_threshold']:
            pred_class = torch.tensor([len(config['label_encoder'].classes_) - 1], device=device)  # Unknown class
        return pred_class
res=inference(student_with_attention)

execution_time_t = timeit.timeit(lambda: inference(student_with_attention), number=1000) / 1000
print("Inference time for Student with attention =", execution_time)
#############################################################################################################################################

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, teacher_outputs, labels):
        # Soft targets loss
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs/self.temperature, dim=1),
            F.softmax(teacher_outputs/self.temperature, dim=1)) * (self.alpha * self.temperature * self.temperature)
        
        # Hard targets loss
        hard_loss = self.criterion(student_outputs, labels) * (1. - self.alpha)
        
        return soft_loss + hard_loss

# Prepare for distillation
teacher_model = model  # Your fine-tuned VGGish model
teacher_model.eval()  # Set teacher to eval mode

# Student model optimizer and scheduler
student_optimizer = optim.AdamW(student_with_attention.parameters(), lr=0.001, weight_decay=0.01)
student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    student_optimizer, mode='max', factor=0.5, patience=3, verbose=True
)
distill_criterion = DistillationLoss(temperature=3.0, alpha=0.7)
num_classes = 20
confidence_threshold = 0.7


def distill_train(student_with_attention, teacher, train_loader, optimizer, criterion, device):
    student_with_attention.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Wrap the train_loader with tqdm for progress bar
    train_iter = tqdm(train_loader, desc="Training", unit="batch", leave=False)
    
    for inputs, additional_features, labels in train_iter:
        inputs = inputs.to(device)
        additional_features = additional_features.to(device)
        labels = labels.to(device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = teacher(inputs, additional_features)
        
        # Student forward pass
        optimizer.zero_grad()
        student_logits = student_with_attention(inputs, additional_features)
        
        # Calculate distillation loss
        loss = criterion(student_logits, teacher_logits, labels)
        
        # Backward pass
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(student_with_attention.parameters(), max_norm=1.0)

        optimizer.step()
        
        running_loss += loss.item()
        probs = torch.softmax(student_logits, dim=1)
        max_probs, predicted = torch.max(probs, 1)
        
        # Apply same confidence threshold
        predicted[max_probs < confidence_threshold] = num_classes-1
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_iter.set_postfix({
            'loss': running_loss/(train_iter.n+1),
            'acc': 100.*correct/total
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Enhanced Distillation validation with tqdm
def distill_validate(student_with_attention, val_loader, device):
    student_with_attention.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Wrap the val_loader with tqdm for progress bar
    val_iter = tqdm(val_loader, desc="Validating", unit="batch", leave=False)
    
    with torch.no_grad():
        for inputs, additional_features, labels in val_iter:
            inputs = inputs.to(device)
            additional_features = additional_features.to(device)
            labels = labels.to(device)
            
            outputs = student_with_attention(inputs, additional_features)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            preds[max_probs < confidence_threshold] = num_classes-1
            
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar description
            val_iter.set_postfix({
                'acc': 100.*correct/total
            })
    
    val_acc = 100. * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    return val_acc, val_f1

# Enhanced Run distillation training with epoch progress bar
best_student_acc = 0.0
num_distill_epochs = 60  # Increased epochs
patience = 7  # For early stopping
patience_counter = 0
warmup_epochs = 3
base_lr = 0.001

# Verify teacher model performance first
teacher_val_acc, teacher_val_f1 = distill_validate(teacher_model, val_loader, device)
print(f"Teacher model validation: Acc {teacher_val_acc:.2f}%, F1 {teacher_val_f1:.4f}")

accuracy_data=[]
# Create epoch progress bar
epoch_iter = tqdm(range(num_distill_epochs), desc="Epochs", unit="epoch")

for epoch in epoch_iter:
    # Learning rate warmup
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in student_optimizer.param_groups:
            param_group['lr'] = lr
    # Train
    train_loss, train_acc = distill_train(student_with_attention, teacher_model, train_loader, student_optimizer, distill_criterion, device)
    
    # Validate
    val_acc, val_f1 = distill_validate(student_with_attention, val_loader, device)
    accuracy_data.append((train_acc,val_acc))
    # Update scheduler
    student_scheduler.step(val_acc)
    
    # Update epoch progress bar description
    current_lr = student_optimizer.param_groups[0]['lr']
    epoch_iter.set_postfix({
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'lr': current_lr
    })
    
    # Save best student
    if val_acc > best_student_acc:
        best_student_acc = val_acc
        torch.save(student_with_attention.state_dict(), "E:/Project_Experiments/only_dolphin_experiment/best_student_with_attention_model.pth")
        tqdm.write(f"Epoch {epoch+1}: Saved new best student model with val_acc {val_acc:.2f}%")

output_train_vs_val="E:/Project_Experiments/only_dolphin_experiment/train_vs_val_student_with_attention.xlsx"
df = pd.DataFrame(accuracy_data, columns=["Train Accuracy", "Validation Accuracy"])
df.to_excel(output_train_vs_val, index=False)
tqdm.write(f"Train vs Validation data written to {output_train_vs_val}")
# Enhanced test_model function with tqdm
def test_model(model, test_loader, label_encoder, test_dataset, device, output_file="E:/Project_Experiments/only_dolphin_experiment/student_with_attention_misclassified_samples.xlsx"):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_samples = []

    # Wrap test_loader with tqdm
    test_iter = tqdm(test_loader, desc="Testing", unit="batch")
    
    # Important: Get the indices from the dataset to map back correctly
    sample_indices = getattr(test_loader.dataset, 'indices', list(range(len(test_loader.dataset))))
    current_idx = 0  # Pointer to track the sample index

    with torch.no_grad():
        for inputs, additional_features, labels in test_iter:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            additional_features = additional_features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, additional_features)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            preds[max_probs < confidence_threshold] = num_classes - 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(batch_size):
                if preds[i] != labels[i]:
                    sample_idx = sample_indices[current_idx + i]  # Map back to original dataset index
                    file_path = test_dataset._get_audio_sample_path(sample_idx)
                    actual_class = label_encoder.inverse_transform([labels[i].item()])[0]
                    classified_to = label_encoder.inverse_transform([preds[i].item()])[0]
                    misclassified_samples.append({
                        "filename": file_path,
                        "actual class": actual_class,
                        "classified to": classified_to
                    })
            current_idx += batch_size

            # Update tqdm
            test_iter.set_postfix({
                'processed': f"{current_idx}/{len(test_loader.dataset)}"
            })

    if misclassified_samples:
        df = pd.DataFrame(misclassified_samples)
        df.to_excel(output_file, index=False)
        tqdm.write(f"Misclassified samples saved to {output_file}")

    # Calculate and visualize confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tqdm.write("Classification Report:\n" + classification_report(all_labels, all_preds))

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Test the student model with progress indication
print("\nTesting best student model...")
student_with_attention.load_state_dict(torch.load("E:/Project_Experiments/only_dolphin_experiment/best_student_with_attention_model.pth"))
#test_model(student, test_loader, train_dataset.label_encoder)
test_model(student_with_attention, test_loader, train_dataset.label_encoder, test_dataset, device)

# Save the final student model
student_save_path = "E:/Project_Experiments/only_dolphin_experiment/final_student_model_with_attention.pth"
torch.save({
    'model_state_dict': student_with_attention.state_dict(),
    'label_encoder_classes': train_dataset.label_encoder.classes_,
    'num_additional_features': num_additional_features,
    'confidence_threshold': confidence_threshold,
    'feature_columns': train_dataset.feature_columns,
    'sample_rate': sample_rate,
    'num_samples': num_samples
}, student_save_path)
print(f"Student model saved to {student_save_path}")


#############################################################################################################################################
# print(model)  # Basic architecture overview

# # # For detailed summary (add after model definition)
# from torchviz import make_dot
# print(model)
# try:
#     #from torchsummary import summary
#     print("\n=== Detailed Model Summary ===")
#     # Note: We need to provide input sizes for both mel_spec and additional_features
#     # Since summary can't handle multiple inputs directly, we'll show them separately
#     print("Mel Spectrogram path summary:")
#     summary(model.features, (1, 96, 64), device=device)
#     print("\nClassifier path summary:")
#     summary(model.fc, (12288 + 18,), device=device)
#     model.eval()
#     #print("Checking the sample...")
#     #print("Extracting mel_spec + Additional features...")
#     mel_spec,additional_features=process_audio_file(audio_sample_path)
#     mel_spec=mel_spec.unsqueeze(0)
#     additional_features=additional_features.unsqueeze(0)
#     with torch.no_grad():
#         output = model(mel_spec.to(device), additional_features.to(device))

#     graph = make_dot(output, params=dict(model.named_parameters()))  # Create computation graph
#     # Save as PNG
#     graph.render("E:/Project_Experiments/only_dolphin_experiment/teacher_model_summary", format="png", cleanup=True)
#     print("\nModel architecture saved as 'teacher_model_summary.png'")
    
# except ImportError:
#     print("torchsummary not available, skipping detailed summary")

# print(student)

# try:
#     print("\n=== Detailed student Summary ===") 
#     print("Mel Spectrogram path summary:")
#     summary(student.cnn, (1, 96, 64), device=device)
#     print("\nClassifier path summary:")
#     summary(student.fc, (64 + 18,), device=device)

#     student.eval()

#     # Extract features
#     mel_spec, additional_features = process_audio_file(audio_sample_path)
#     mel_spec = mel_spec.unsqueeze(0)
#     additional_features = additional_features.unsqueeze(0)

#     with torch.no_grad():
#         output = student(mel_spec.to(device), additional_features.to(device))

#     # Generate computation graph
#     graph = make_dot(output, params=dict(student.named_parameters()))

#     # Fix 1: Explicitly set PNG format
#     graph.format = "png"

#     # Fix 2: Increase graph size (adjust if needed)
#     graph.attr(size="12,12")  # Change to larger values if still cropped

#     # Fix 3: Use a safe path (avoid spaces)
#     save_path = "E:/Project_Experiments/only_dolphin_experiment/student_model_summary"

#     graph.render(save_path, cleanup=True)
#     print(f"\nModel architecture saved as '{save_path}.png'")

# except ImportError:
#     print("torchsummary not available, skipping detailed summary")