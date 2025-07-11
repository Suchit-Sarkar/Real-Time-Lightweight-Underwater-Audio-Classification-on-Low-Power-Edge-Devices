import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from scipy import signal
import random
import os
from tqdm import tqdm

class MarineMammalAugmenter:
    
    def __init__(self, duration_seconds=5):
        self.duration_seconds = duration_seconds
        
    def load_audio(self, file_path):
        try:
            waveform, sr = torchaudio.load(file_path)
            self.sample_rate = sr  # Keep track of the original sample rate
            self.duration_samples = int(sr * self.duration_seconds)
                
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
    
    def time_shift(self, waveform, shift_limit=0.3):

        shift_factor = random.uniform(-shift_limit, shift_limit)
        shift_amount = int(waveform.shape[1] * shift_factor)
        return torch.roll(waveform, shifts=shift_amount, dims=1)
    
    def add_ambient_noise(self, waveform, noise_factor_range=(0.001, 0.01)):

        noise_factor = random.uniform(*noise_factor_range)
        
        # Generate pink noise (1/f noise common in ocean environments)
        noise_length = waveform.shape[1]
        noise = np.random.randn(noise_length)
        # Get the power spectrum
        _, psd = signal.welch(noise, fs=self.sample_rate, nperseg=min(noise_length, 1024))
        noise = torch.from_numpy(psd[:noise_length] if len(psd) >= noise_length else 
                               np.pad(psd, (0, noise_length - len(psd)))).float()
        
        # Reshape noise to match waveform
        noise = noise.unsqueeze(0)
        if noise.shape[1] < waveform.shape[1]:
            noise = noise.repeat(1, (waveform.shape[1] // noise.shape[1]) + 1)
        noise = noise[:, :waveform.shape[1]]
        
        # Add noise to original signal
        return waveform + noise_factor * noise
    
    def pitch_shift(self, waveform, pitch_shift_range=(-1, 1)):

        try:
            # Convert to numpy for librosa processing
            waveform_np = waveform.squeeze().numpy()
            
            # Apply pitch shift
            semitones = random.uniform(*pitch_shift_range)
            pitched = librosa.effects.pitch_shift(
                waveform_np, 
                sr=self.sample_rate, 
                n_steps=semitones
            )
            
            return torch.tensor(pitched).unsqueeze(0)
        except Exception as e:
            print(f"Error in pitch_shift: {e}")
            return waveform
    
    def time_stretch(self, waveform, stretch_range=(0.9, 1.1)):

        try:
            # Convert to numpy for librosa processing
            waveform_np = waveform.squeeze().numpy()
            
            # Apply time stretch
            rate = random.uniform(*stretch_range)
            stretched = librosa.effects.time_stretch(waveform_np, rate=rate)
            
            # Ensure output length is consistent
            if len(stretched) > self.duration_samples:
                stretched = stretched[:self.duration_samples]
            elif len(stretched) < self.duration_samples:
                # Pad if needed
                padding = np.zeros(self.duration_samples - len(stretched))
                stretched = np.concatenate([stretched, padding])
            
            return torch.tensor(stretched).unsqueeze(0)
        except Exception as e:
            print(f"Error in time_stretch: {e}")
            return waveform
    
    def apply_band_pass_filter(self, waveform, min_freq=1000, max_freq=24000):

        try:
            waveform_np = waveform.squeeze().numpy()
            
            # Create band-pass filter
            nyquist = self.sample_rate // 2
            low = min_freq / nyquist
            high = min(max_freq / nyquist, 0.99)  # Ensure high is less than 1.0
            
            # Get filter coefficients
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, waveform_np)
            #tensor_signal = torch.tensor(filtered.copy(), dtype=torch.float32).unsqueeze(0)

            return torch.tensor(filtered).unsqueeze(0)
        except Exception as e:
            print(f"Error in band_pass_filter: {e}")
            return waveform
    
    def apply_dynamic_range_compression(self, waveform, threshold_db=-20, ratio=2):

        try:
            # Convert to dB scale
            amplitude_to_db = T.AmplitudeToDB()
            db_to_amplitude = T.DB2Amplitude()
            
            waveform_db = amplitude_to_db(waveform.abs() + 1e-9)  # Add small value to avoid log(0)
            
            # Apply compression: values above threshold are attenuated by ratio
            mask = waveform_db > threshold_db
            waveform_db[mask] = threshold_db + (waveform_db[mask] - threshold_db) / ratio
            
            # Convert back to linear scale
            compressed = db_to_amplitude(waveform_db) * torch.sign(waveform)
            
            return compressed
        except Exception as e:
            print(f"Error in dynamic_range_compression: {e}")
            return waveform
    
    def selective_frequency_masking(self, waveform, mask_param=(80, 600)):

        try:
            # Convert to spectrogram
            spec_transform = T.Spectrogram(n_fft=1024, hop_length=512)
            griffin_lim_transform = T.GriffinLim(n_fft=1024, hop_length=512)
            
            spec = spec_transform(waveform)
            
            # Apply frequency masking
            mask_param_val = random.randint(*mask_param)
            freq_mask = T.FrequencyMasking(freq_mask_param=mask_param_val)
            masked_spec = freq_mask(spec)
            
            # Convert back to waveform using Griffin-Lim algorithm
            griffin_lim = griffin_lim_transform(masked_spec)
            
            # Ensure same length as original
            if len(griffin_lim.shape) > 1:
                griffin_lim = griffin_lim.squeeze(0)
                
            if griffin_lim.shape[0] > waveform.shape[1]:
                griffin_lim = griffin_lim[:waveform.shape[1]]
            elif griffin_lim.shape[0] < waveform.shape[1]:
                padding = torch.zeros(waveform.shape[1] - griffin_lim.shape[0])
                griffin_lim = torch.cat([griffin_lim, padding])
                
            return griffin_lim.unsqueeze(0)
        except Exception as e:
            print(f"Error in selective_frequency_masking: {e}")
            return waveform
    
    def simulate_water_surface_reflection(self, waveform, delay_range=(0.01, 0.03), attenuation=0.6):

        try:
            delay_time = random.uniform(*delay_range)
            delay_samples = int(delay_time * self.sample_rate)
            
            if delay_samples >= waveform.shape[1]:
                delay_samples = waveform.shape[1] // 10  # Set to 10% of signal length if too large
            
            # Create delayed version
            delayed = torch.zeros_like(waveform)
            delayed[:, delay_samples:] = waveform[:, :-delay_samples] * attenuation
            
            # Add to original
            result = waveform + delayed
            
            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(result))
            if max_val > 0:  # Avoid division by zero
                result = result / max_val
            
            return result
        except Exception as e:
            print(f"Error in water_surface_reflection: {e}")
            return waveform
    
    def add_varying_snr_noise(self, waveform, snr_range=(15, 25)):

        try:
            # Calculate signal power
            signal_power = torch.mean(waveform ** 2)
            
            # Generate random SNR from range
            snr = random.uniform(*snr_range)
            
            # Calculate noise power based on SNR
            noise_power = signal_power / (10 ** (snr / 10))
            
            if torch.isnan(noise_power) or torch.isinf(noise_power) or noise_power <= 0:
                return waveform  # Skip if we get invalid values
                
            # Generate noise
            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
            
            # Add noise to signal
            return waveform + noise
        except Exception as e:
            print(f"Error in add_varying_snr_noise: {e}")
            return waveform
    
    def cyclic_frequency_shift(self, waveform, shift_range=(100, 300)):

        try:
            # Convert to spectrogram
            spec_transform = T.Spectrogram(n_fft=1024, hop_length=512)
            griffin_lim_transform = T.GriffinLim(n_fft=1024, hop_length=512)
            
            spec = spec_transform(waveform)
            
            # Get shift amount in bins
            shift_amount = random.randint(*shift_range)
            
            if spec.shape[1] <= shift_amount:
                shift_amount = spec.shape[1] // 10  # Limit to 10% of the bins if too large
            
            # Cyclic shift in frequency domain
            shifted_spec = torch.roll(spec, shifts=shift_amount, dims=1)
            
            # Convert back to time domain
            griffin_lim = griffin_lim_transform(shifted_spec)
            
            # Ensure same length
            if len(griffin_lim.shape) > 1:
                griffin_lim = griffin_lim.squeeze(0)
                
            if griffin_lim.shape[0] > waveform.shape[1]:
                griffin_lim = griffin_lim[:waveform.shape[1]]
            elif griffin_lim.shape[0] < waveform.shape[1]:
                padding = torch.zeros(waveform.shape[1] - griffin_lim.shape[0])
                griffin_lim = torch.cat([griffin_lim, padding])
                
            return griffin_lim.unsqueeze(0)
        except Exception as e:
            print(f"Error in cyclic_frequency_shift: {e}")
            return waveform
    
    def augment(self, waveform, techniques=None, probabilities=None):

        if waveform is None:
            return None
            
        if techniques is None:
            # Default techniques with varying preservation levels
            techniques = [
                self.time_shift,              # High preservation
                self.add_ambient_noise,       # High preservation
                self.pitch_shift,             # Medium preservation
                self.time_stretch,            # Medium preservation
                self.apply_band_pass_filter,  # High preservation
                self.apply_dynamic_range_compression,  # Medium preservation
                self.selective_frequency_masking,      # Medium preservation
                self.simulate_water_surface_reflection,  # Medium-high preservation
                self.add_varying_snr_noise,   # High preservation
                self.cyclic_frequency_shift   # Medium preservation
            ]
        
        if probabilities is None:
            # Higher probabilities for techniques that better preserve features
            probabilities = [0.8, 0.7, 0.5, 0.5, 0.6, 0.4, 0.3, 0.6, 0.7, 0.4]
            
        # Normalize probabilities
        probabilities = [p / sum(probabilities) for p in probabilities]
        
        # Make a copy of the original waveform
        augmented = waveform.clone()
        
        # Randomly apply augmentations based on their probabilities
        for technique, prob in zip(techniques, probabilities):
            if random.random() < prob:
                try:
                    augmented = technique(augmented)
                except Exception as e:
                    print(f"Error applying technique {technique.__name__}: {e}")
                    # Continue with unmodified waveform for this technique
                
        return augmented
    
    def batch_augment(self, input_dir, output_dir, num_augmentations=5, species_subfolders=True):

        print(f"Starting batch augmentation from {input_dir} to {output_dir}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist")
            return
            
        # Check if we can read the input directory
        try:
            test_list = os.listdir(input_dir)
            print(f"Found {len(test_list)} items in input directory")
        except Exception as e:
            print(f"Error accessing input directory: {e}")
            return
        
        if species_subfolders:
            # Process each species folder separately
            for species_folder in os.listdir(input_dir):
                species_path = os.path.join(input_dir, species_folder)
                
                if os.path.isdir(species_path):
                    # Create corresponding output folder
                    species_output_path = os.path.join(output_dir, species_folder)
                    os.makedirs(species_output_path, exist_ok=True)
                    
                    # Get all WAV files for this species
                    audio_files = [f for f in os.listdir(species_path) if f.endswith('.wav')]
                    
                    if not audio_files:
                        print(f"No WAV files found in {species_path}")
                        continue
                        
                    print(f"Found {len(audio_files)} WAV files in {species_path}")
                    
                    for file_name in tqdm(audio_files, desc=f"Augmenting {species_folder}"):
                        file_path = os.path.join(species_path, file_name)
                        base_name = os.path.splitext(file_name)[0]
                        
                        # Load original audio
                        waveform = self.load_audio(file_path)
                        if waveform is None:
                            continue
                        
                        # Create augmented versions
                        for i in range(num_augmentations):
                            try:
                                augmented = self.augment(waveform)
                                
                                # Save augmented file
                                output_path = os.path.join(species_output_path, f"{base_name}_aug_{i+1}.wav")
                                torchaudio.save(output_path, augmented, self.sample_rate)
                                print(f"Saved: {output_path}")
                            except Exception as e:
                                print(f"Error augmenting {file_path} (version {i+1}): {e}")
        else:
            # Flat directory structure
            audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
            
            if not audio_files:
                print(f"No WAV files found in {input_dir}")
                return
                
            print(f"Found {len(audio_files)} WAV files in {input_dir}")
            
            for file_name in tqdm(audio_files, desc="Augmenting audio files"):
                file_path = os.path.join(input_dir, file_name)
                base_name = os.path.splitext(file_name)[0]
                
                # Load original audio
                waveform = self.load_audio(file_path)
                if waveform is None:
                    continue
                
                # Create augmented versions
                for i in range(num_augmentations):
                    try:
                        augmented = self.augment(waveform)
                        
                        # Save augmented file
                        output_path = os.path.join(output_dir, f"{base_name}_aug_{i+1}.wav")
                        torchaudio.save(output_path, augmented, self.sample_rate)
                        #print(f"Saved: {output_path}")
                    except Exception as e:
                        print(f"Error augmenting {file_path} (version {i+1}): {e}")
                
        print(f"Augmentation complete. Output saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Initialize augmenter
    augmenter = MarineMammalAugmenter(duration_seconds=5)
    # Example directory structure
    base_directory = "E:/Project_Experiments/only_dolphin_experiment/separated_augmented/validate_data/noise"
    output_directory = "E:/Project_Experiments/only_dolphin_experiment/separated_augmented/validate_data/noise_augmented"
    
    # Check if the path exists
    print(f"Checking if base directory exists: {os.path.exists(base_directory)}")
    
    augmenter.batch_augment(base_directory, output_directory, num_augmentations=15, species_subfolders=False)
    