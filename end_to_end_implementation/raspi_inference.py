import time
import numpy as np
import torchaudio
import librosa
#import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import scipy.signal as signal
import timeit
import queue
import threading
import pyaudio
import wave
from collections import deque
import matplotlib.pyplot as plt
import os
# Global variables
latest_result = {"class": None, "label": "Initializing...", "score": 0.0}
audio_buffer = deque(maxlen=80000)  # 5 seconds at 16kHz
buffer_lock = threading.Lock()
recording_active = False
recording_thread = None
latest_spectrogram = None
sample_rate=16000
num_samples = sample_rate * 5

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
sample_rate = 16000  # Required sample rate for VGGish
num_samples = sample_rate * 5  # 5 seconds of audio at 16 kHz
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,  # Smaller window size for VGGish
        hop_length=160,  # Matches VGGish expectation
        n_mels=64
    )
  

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
    try:
        # Return zeros if no signal
        if signal is None or len(signal) == 0:
            return np.zeros(18, dtype=np.float32)
            
        # Ensure proper shape
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=0)
        
        signal = signal.astype(np.float32)
        # Rest of your feature extraction...
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(18, dtype=np.float32)
    
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

def process_audio(audio_data,sr=16000): 
    global latest_spectrogram, num_samples
    print(audio_data.shape)
    try:
        if audio_data is None or len(audio_data) == 0:
            print("No audio data\n")
            return None, None
        # Ensure we have a numpy array
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)
        if len(audio_data) < 1:
            print("No audio data\n")
            return None, None
        
        print("Before calling process_audio() right after recording\n")
        print(audio_data.shape,audio_data.dtype,audio_data.ndim)
        
        signal = torch.tensor(audio_data, dtype=torch.float32)
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
        mel_spec = torch.log(mel_spec + 1e-6)  # Convert to log scale

        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(96, 64), mode='bilinear', align_corners=False
        ).squeeze(0)
        mel_spec = mel_spec.to(device)
        print(mel_spec)
        #save_spectrogram_image(mel_spec)
        # Ensure 3D shape
        mel_spec = mel_spec.unsqueeze(0) if mel_spec.ndim == 2 else mel_spec
        latest_spectrogram=mel_spec
        if mel_spec is None:
            return None, None
        return mel_spec, additional_features
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None,None

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
def save_spectrogram_image(tensor, filename='static/spectrogram.png'):
    if tensor is None:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    spec = tensor.squeeze().cpu().numpy()
    plt.figure(figsize=(4, 3))
    plt.imshow(spec, origin='lower', aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
 
device="cpu" 
num_additional_features=18
student_with_attention = AudioTransformerStudent(num_additional_features,20).to(device)
model_path="E:/Project_Experiments/only_dolphin_experiment/final_student_model_with_attention_no_kd.pth"
# mel_spec_cpu,additional_features_cpu=extract_feature(audio_sample_path) # Load with weights_only=False since we saved numpy arrays
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

student_with_attention.load_state_dict(checkpoint['model_state_dict'])
student_with_attention.eval()

# Reconstruct label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = checkpoint['label_encoder_classes']


def inference():
    global latest_result, audio_buffer

    if student_with_attention is None or label_encoder is None:
        return {"class": None, "label": "Model not loaded", "score": 0.0}
    
    try:
        with buffer_lock:
            if len(audio_buffer) < sample_rate:  # Ensure we have enough audio
                return {"class": None, "label": "Not enough audio data", "score": 0.0}
            
            # Convert deque to numpy array
            audio_data = np.array(list(audio_buffer), dtype=np.float32)
            # # Ensure we have at least 5 seconds of audio
            # if len(audio_data) < sample_rate * 5:
            #     pad_length = sample_rate * 5 - len(audio_data)
            #     audio_data = np.pad(audio_data, (0, pad_length), 'constant')
            
            # # If we have more than 5 seconds, take the last 5 seconds
            # if len(audio_data) > sample_rate * 5:
            #     audio_data = audio_data[-sample_rate * 5:]
            audio_data = audio_data.reshape(1, -1)

            mel_spec, additional_features = process_audio(audio_data,16000)
            print("Mel_spec_shape",mel_spec.shape,"Add_feat_shape",additional_features.shape)
        
            if mel_spec is None or additional_features is None:
                return {"class": None, "label": "Error processing audio", "score": 0.0}
        
        # Add batch dimension if not present
            additional_features = additional_features.unsqueeze(0)
        
        # Move to device
            mel_spec = mel_spec.to(device)
            additional_features = additional_features.to(device)
            mel_spec=mel_spec.unsqueeze(0)
        
            with torch.no_grad():
                output = student_with_attention(mel_spec, additional_features)
                probs = torch.softmax(output, dim=1)
                max_prob, pred_class = torch.max(probs, 1)
    
        # Apply confidence threshold
                if max_prob < checkpoint['confidence_threshold']+0.1:
                    pred_class = torch.tensor([len(label_encoder.classes_) - 1], device=device)  # Unknown class

                pred_label = label_encoder.inverse_transform(pred_class.cpu().numpy())
                # Apply confidence threshold
        
                print({
                "class":pred_class.item(),
                "label": pred_label,
                "score": max_prob.item()
                })
                latest_result={
                "class":pred_class.item(),
                "label": pred_label.item(),
                "score": max_prob.item()
                }
                return latest_result
    except Exception as e:
        print(f"Inference error: {e}")
        return {"class": None, "label": f"Error: {str(e)}", "score": 0.0}


def get_latest_result():
    global latest_result
    return latest_result

def get_latest_spectrogram():
    global latest_spectrogram
    if latest_spectrogram is not None:
        if torch.is_tensor(latest_spectrogram):      
            latest_spectrogram = latest_spectrogram.squeeze().cpu().numpy()
        if latest_spectrogram.ndim==3:
            latest_spectrogram=latest_spectrogram[0]
        latest_spectrogram = latest_spectrogram.tolist()
        return {
            "data": latest_spectrogram,
            "min": float(np.min(latest_spectrogram)),
            "max": float(np.max(latest_spectrogram))
        }
    return {"data": [], "min": 0, "max": 1}


def audio_callback(in_data, frame_count, time_info, status):

    global audio_buffer,buffer_lock
    
    if not recording_active:
        return (in_data, pyaudio.paContinue)
    
    # Convert byte data to numpy array (assuming 16-bit PCM)
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Add to buffer
    with buffer_lock:
        audio_buffer.extend(audio_data)
    
    return (in_data, pyaudio.paContinue)

def start_recording():

    global recording_active, recording_thread, audio_buffer
    
    if recording_active:
        return

    recording_active = True
    # Clear buffer
    with buffer_lock:
        audio_buffer.clear()
    
    # Start recording thread
    recording_thread = threading.Thread(target=recording_worker)
    recording_thread.daemon = True
    recording_thread.start()
    
    print("Recording started")

def stop_recording():
    global recording_active
    
    if not recording_active:
        return
    
    recording_active = False
    
    # Wait for thread to finish
    if recording_thread:
        recording_thread.join(timeout=1.0)
    print("Recording stopped")

def recording_worker():
    global recording_active
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024,
                        stream_callback=audio_callback)
        
        # Start stream
        stream.start_stream()
        
        # Keep running while recording is active
        while recording_active and stream.is_active():
            time.sleep(0.1)
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        
        # Terminate PyAudio
        p.terminate()
    
    except Exception as e:
        print(f"Recording error: {e}")
        recording_active = False

def inference_worker(interval=1.0):
    while True:
        try:
            if recording_active:
                inference()
            time.sleep(interval)
        except Exception as e:
            print(f"Inference worker error: {e}")
            time.sleep(1)

# Initialize recording and inference threads
def start_inference_thread(interval=5.0):
    inference_thread = threading.Thread(target=inference_worker, args=(interval,))
    inference_thread.daemon = True
    inference_thread.start()

if __name__ != "__main__":
    start_inference_thread()

