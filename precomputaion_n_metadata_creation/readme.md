# Dolphin Species Classification - Audio Metadata Extraction

## Overview

This project provides a comprehensive audio feature extraction pipeline for dolphin species classification using machine learning. The system processes audio files from 20 different dolphin species and extracts normalized acoustic features suitable for training classification models.

## Features

- **Multi-species Support**: Handles 20 dolphin species plus noise classification
- **Comprehensive Feature Extraction**: Extracts 18 acoustic features including spectral and temporal characteristics
- **Audio Preprocessing**: Applies bandpass filtering, resampling, and normalization
- **Batch Processing**: Processes entire directories of audio files with progress tracking
- **Normalized Output**: Features are normalized using predefined scaling factors for optimal model performance

## Requirements

### Python Dependencies

```bash
pip install pandas numpy torchaudio librosa torch scipy tqdm openpyxl
```

### Detailed Requirements

- **pandas**: Data manipulation and Excel export
- **numpy**: Numerical computing and array operations
- **torchaudio**: Audio loading and tensor operations
- **librosa**: Advanced audio feature extraction
- **torch**: PyTorch tensor operations and neural network support
- **scipy**: Signal processing (bandpass filtering)
- **tqdm**: Progress bar visualization
- **openpyxl**: Excel file writing engine

### System Requirements

- **Python**: 3.7+
- **Memory**: 8GB RAM minimum (16GB recommended for large datasets)
- **Storage**: Sufficient space for audio files and output metadata
- **Audio Formats**: Supports WAV files

## Project Structure

```
project/
├── validate_metadata_2.py    # Validation set metadata extraction
├── train_metadata_2.py       # Training set metadata extraction  
├── test_metadata_2.py        # Test set metadata extraction
├── README.md                  # This file
└── data/
    ├── validate_data/         # Validation audio files
    ├── train_data/           # Training audio files
    └── test_data/            # Test audio files
        ├── AtlanticSpottedDolphin/
        ├── BottlenoseDolphin/
        ├── Boutu_AmazonRiverDolphin/
        └── ... (other species folders)
```

## Supported Species

The system supports classification of the following 20 dolphin species:

| Species Name | Numeric Label |
|-------------|---------------|
| AtlanticSpottedDolphin | 0 |
| BottlenoseDolphin | 1 |
| Boutu_AmazonRiverDolphin | 2 |
| ClymeneDolphin | 3 |
| Commerson'sDolphin | 4 |
| CommonDolphin | 5 |
| DuskyDolphin | 6 |
| Fraser'sDolphin | 7 |
| Grampus_Risso'sDolphin | 8 |
| Heaviside'sDolphin | 9 |
| IrawaddyDolphin | 10 |
| noise | 11 |
| LongBeaked(Pacific)CommonDolphin | 12 |
| PantropicalSpottedDolphin | 13 |
| Rough_ToothedDolphin | 14 |
| SpinnerDolphin | 15 |
| StripedDolphin | 16 |
| TucuxiDolphin | 17 |
| White_beakedDolphin | 18 |
| White_sidedDolphin | 19 |

## Extracted Features

The system extracts 18 acoustic features from each audio file:

### Spectral Features
- **Spectral Centroid**: Center of mass of the spectrum
- **Spectral Rolloff**: Frequency below which 85% of energy is concentrated
- **Band Energy Ratio**: Ratio of energy in specific frequency band to total energy

### Temporal Features
- **RMS Energy**: Root mean square energy of the signal
- **Zero Crossing Rate**: Rate of sign changes in the signal

### Cepstral Features
- **MFCCs 1-13**: Mel-frequency cepstral coefficients capturing timbral characteristics

## Audio Processing Pipeline

### 1. Audio Loading and Preprocessing
```python
# Load audio with original sample rate
signal, sr = torchaudio.load(file_path)

# Resample to 96kHz for consistency
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=96000)
signal = resampler(signal)
```

### 2. Signal Conditioning
- **Mono Conversion**: Multi-channel audio converted to mono
- **Duration Standardization**: All clips processed to 5 seconds (480,000 samples at 96kHz)
- **Padding/Trimming**: Shorter clips padded, longer clips trimmed

### 3. Bandpass Filtering
```python
# Apply bandpass filter (1kHz - 24kHz)
filtered_audio = _bandpass_filter(signal, sr, lowcut=1000, highcut=24000)
```

### 4. Feature Extraction
- Spectral and temporal features extracted using librosa
- MFCCs computed with 13 coefficients
- All features converted to 32-bit float precision

### 5. Normalization
Features are normalized using predefined scaling factors:
- Spectral Centroid: ÷ 1000
- RMS Energy: ÷ 1.0
- Zero Crossing Rate: ÷ 0.01
- Spectral Rolloff: ÷ 1000
- Band Energy Ratio: ÷ 1.0
- MFCCs: Various scaling factors (2-500)

## Usage

### Basic Usage

1. **Setup Directory Structure**:
   ```bash
   mkdir -p data/validate_data
   # Organize audio files by species in separate folders
   ```

2. **Update Base Directory**:
   ```python
   base_directory = "path/to/your/audio/data"
   ```

3. **Run Feature Extraction**:
   ```bash
   python validate_metadata_2.py
   ```

### For Complete Pipeline

```bash
# Extract features for all datasets
python train_metadata_2.py      # Training set
python validate_metadata_2.py   # Validation set
python test_metadata_2.py       # Test set
```

## Output Format

The script generates an Excel file (`validate_metadata.xlsx`) with the following columns:

| Column | Description |
|--------|-------------|
| file_name | Original audio filename |
| species_name | Species name string |
| numeric_value | Species numeric label (0-19) |
| sample_rate | Original sample rate |
| spectral_centroid | Normalized spectral centroid |
| rms_energy | Normalized RMS energy |
| zero_crossing_rate | Normalized zero crossing rate |
| spectral_rolloff | Normalized spectral rolloff |
| band_energy_ratio | Normalized band energy ratio |
| mfcc_1 to mfcc_13 | Normalized MFCC coefficients |

## Configuration

### Hyperparameters

```python
# Audio processing parameters
TARGET_SAMPLE_RATE = 96000      # Target sample rate (Hz)
AUDIO_DURATION = 5.0            # Duration in seconds
BANDPASS_LOW = 1000             # Bandpass filter low cutoff (Hz)
BANDPASS_HIGH = 24000           # Bandpass filter high cutoff (Hz)
N_MFCC = 13                     # Number of MFCC coefficients
```

### Normalization Factors
The normalization factors are optimized based on statistical analysis of the dataset:

```python
norm_factors = [
    1000.0,  # spectral_centroid
    1.0,     # rms_energy
    0.01,    # zero_crossing_rate
    # ... (see code for complete list)
]
```

## Performance Considerations

- **Memory Usage**: ~100MB per 1000 audio files
- **Processing Speed**: ~2-5 seconds per audio file
- **Parallel Processing**: Can be modified to use multiprocessing for faster batch processing

## Error Handling

The system includes comprehensive error handling:
- Invalid audio files are skipped with error logging
- Missing directories are reported
- Filter stability is checked before application
- Feature extraction failures are caught and logged

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or increase system RAM
2. **File Not Found**: Check directory structure and file permissions
3. **Audio Loading Error**: Verify audio file format and integrity
4. **Filter Instability**: Check sample rate compatibility

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is designed for research and educational purposes in marine bioacoustics and machine learning.

## Contributing

When contributing to this project:
1. Maintain the existing feature extraction pipeline
2. Test with sample audio files before processing large datasets
3. Document any changes to normalization factors
4. Ensure compatibility with the existing ML pipeline

## References

- Librosa: Audio analysis library
- PyTorch: Deep learning framework
- Scipy: Scientific computing library

For questions or issues, please refer to the documentation of the respective libraries used in this project.