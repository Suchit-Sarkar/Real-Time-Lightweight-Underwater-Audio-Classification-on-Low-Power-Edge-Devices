# Marine Mammal Audio Data Augmentation

A comprehensive Python library for augmenting marine mammal audio data using various signal processing techniques. This tool is designed to enhance machine learning datasets by generating realistic variations of marine mammal vocalizations while preserving their essential acoustic characteristics.

## Features

- **10 Advanced Augmentation Techniques**: Time shifting, ambient noise addition, pitch shifting, time stretching, band-pass filtering, dynamic range compression, frequency masking, water surface reflection simulation, SNR noise addition, and cyclic frequency shifting
- **Batch Processing**: Process entire directories with support for species-specific folder structures
- **Configurable Parameters**: Adjustable probability weights and technique parameters
- **Marine Environment Specific**: Techniques tailored for underwater acoustic environments
- **Robust Error Handling**: Comprehensive error handling and logging

## Requirements

### Python Dependencies

```bash
pip install numpy torch torchaudio librosa scipy tqdm
```

### Detailed Requirements

```
numpy>=1.21.0
torch>=1.9.0
torchaudio>=0.9.0
librosa>=0.8.1
scipy>=1.7.0
tqdm>=4.62.0
```

### System Requirements

- Python 3.7+
- Audio files in WAV format
- Sufficient disk space for augmented output files

## Installation

1. Clone or download the repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Import the `MarineMammalAugmenter` class in your Python script

## Quick Start

```python
from data_augmentation import MarineMammalAugmenter

# Initialize the augmenter
augmenter = MarineMammalAugmenter(duration_seconds=5)

# Augment a single audio file
waveform = augmenter.load_audio("path/to/audio.wav")
augmented_waveform = augmenter.augment(waveform)

# Batch augment entire directory
augmenter.batch_augment(
    input_dir="path/to/input/directory",
    output_dir="path/to/output/directory",
    num_augmentations=5,
    species_subfolders=True
)
```

## Augmentation Techniques

### 1. Time Shift
- **Purpose**: Shifts audio signal in time domain
- **Parameters**: `shift_limit=0.3` (30% of signal length)
- **Preservation**: High - maintains frequency characteristics

### 2. Ambient Noise Addition
- **Purpose**: Adds realistic ocean background noise (pink noise)
- **Parameters**: `noise_factor_range=(0.001, 0.01)`
- **Preservation**: High - simulates natural recording conditions

### 3. Pitch Shift
- **Purpose**: Alters fundamental frequency while preserving timing
- **Parameters**: `pitch_shift_range=(-1, 1)` semitones
- **Preservation**: Medium - changes pitch but maintains temporal structure

### 4. Time Stretch
- **Purpose**: Changes playback speed without affecting pitch
- **Parameters**: `stretch_range=(0.9, 1.1)`
- **Preservation**: Medium - maintains pitch but alters timing

### 5. Band-Pass Filter
- **Purpose**: Filters frequencies outside marine mammal vocal range
- **Parameters**: `min_freq=1000Hz`, `max_freq=24000Hz`
- **Preservation**: High - removes irrelevant frequencies

### 6. Dynamic Range Compression
- **Purpose**: Reduces dynamic range for consistent amplitude
- **Parameters**: `threshold_db=-20`, `ratio=2`
- **Preservation**: Medium - maintains frequency content

### 7. Selective Frequency Masking
- **Purpose**: Masks random frequency bands to simulate interference
- **Parameters**: `mask_param=(80, 600)` frequency bins
- **Preservation**: Medium - simulates real-world acoustic interference

### 8. Water Surface Reflection
- **Purpose**: Simulates multipath propagation in water
- **Parameters**: `delay_range=(0.01, 0.03)s`, `attenuation=0.6`
- **Preservation**: Medium-High - adds realistic underwater acoustics

### 9. Varying SNR Noise
- **Purpose**: Adds white noise with varying signal-to-noise ratios
- **Parameters**: `snr_range=(15, 25)` dB
- **Preservation**: High - simulates recording conditions

### 10. Cyclic Frequency Shift
- **Purpose**: Shifts frequency components cyclically
- **Parameters**: `shift_range=(100, 300)` frequency bins
- **Preservation**: Medium - alters frequency relationships

## Hyperparameter Optimization

### Probability Weights

The augmentation techniques use carefully tuned probability weights to balance data diversity with feature preservation:

```python
# Default probabilities (normalized)
probabilities = [
    0.8,  # time_shift - High preservation
    0.7,  # add_ambient_noise - High preservation  
    0.5,  # pitch_shift - Medium preservation
    0.5,  # time_stretch - Medium preservation
    0.6,  # apply_band_pass_filter - High preservation
    0.4,  # apply_dynamic_range_compression - Medium preservation
    0.3,  # selective_frequency_masking - Medium preservation
    0.6,  # simulate_water_surface_reflection - Medium-high preservation
    0.7,  # add_varying_snr_noise - High preservation
    0.4   # cyclic_frequency_shift - Medium preservation
]
```

### Technique Selection Strategy

- **High Preservation Techniques** (0.6-0.8 probability): Maintain essential acoustic features
- **Medium Preservation Techniques** (0.3-0.5 probability): Add variation while preserving most characteristics
- **Stochastic Application**: Each technique is applied independently based on its probability

### Customizable Parameters

Each technique can be customized by modifying its parameters:

```python
# Example: Custom pitch shift range
augmented = augmenter.pitch_shift(waveform, pitch_shift_range=(-2, 2))

# Example: Custom noise levels
augmented = augmenter.add_ambient_noise(waveform, noise_factor_range=(0.005, 0.02))
```

## Usage Examples

### Basic Single File Augmentation

```python
augmenter = MarineMammalAugmenter(duration_seconds=5)
waveform = augmenter.load_audio("dolphin_call.wav")
augmented = augmenter.augment(waveform)
```

### Custom Technique Selection

```python
# Apply only specific techniques
techniques = [augmenter.time_shift, augmenter.add_ambient_noise]
probabilities = [0.8, 0.6]

augmented = augmenter.augment(waveform, techniques=techniques, probabilities=probabilities)
```

### Batch Processing with Species Folders

```python
# Directory structure:
# input_dir/
#   ├── dolphin/
#   │   ├── call1.wav
#   │   └── call2.wav
#   └── whale/
#       ├── song1.wav
#       └── song2.wav

augmenter.batch_augment(
    input_dir="marine_mammals/",
    output_dir="augmented_data/",
    num_augmentations=10,
    species_subfolders=True
)
```

### Flat Directory Processing

```python
# For flat directory structure (all files in one folder)
augmenter.batch_augment(
    input_dir="all_recordings/",
    output_dir="augmented_recordings/",
    num_augmentations=5,
    species_subfolders=False
)
```

## Output Format

- **File naming**: `{original_name}_aug_{number}.wav`
- **Audio format**: WAV files with original sample rate preserved
- **Directory structure**: Maintains input directory organization

## Error Handling

The library includes comprehensive error handling:

- **File loading errors**: Skips corrupted files and continues processing
- **Technique application errors**: Falls back to original waveform if technique fails
- **Directory access errors**: Provides clear error messages for missing directories
- **Parameter validation**: Checks for valid parameter ranges

## Performance Considerations

- **Memory usage**: Processes files individually to manage memory
- **Processing time**: Time varies by technique complexity and file length
- **Disk space**: Output directory requires 5-15x original data size depending on augmentation count

## Tips for Best Results

1. **Start with fewer augmentations** (3-5) and increase based on model performance
2. **Use species-specific folders** for better organization
3. **Monitor disk space** before running large batch operations
4. **Test individual techniques** on sample files before batch processing
5. **Adjust probabilities** based on your specific use case and data characteristics

## Troubleshooting

### Common Issues

1. **"No WAV files found"**: Ensure input directory contains .wav files
2. **Memory errors**: Reduce batch size or duration_seconds parameter
3. **Slow processing**: Consider reducing num_augmentations or using fewer techniques
4. **Output quality issues**: Adjust technique parameters or probabilities

### Debug Mode

Enable verbose output by checking the console messages during batch processing.

## License

This project is designed for research and educational purposes in marine mammal acoustic analysis.

## Contributing

Feel free to contribute additional augmentation techniques or improvements to existing methods.

## Citation

If you use this library in your research, please cite appropriately and acknowledge the specific techniques employed in your data augmentation pipeline.