# Marine Mammal Classification with VGGish and Optuna Hyperparameter Optimization

This project implements a marine mammal species classification system using a pre-trained VGGish model with hyperparameter optimization via Optuna. The system classifies audio recordings of marine mammals into 9 different species categories.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Code Architecture](#code-architecture)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Usage](#usage)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)

## Overview

The project uses a transfer learning approach with VGGish (a pre-trained audio classification model) as the feature extractor, combined with a custom fully connected classifier. Optuna is used for automated hyperparameter tuning to optimize model performance.

### Key Features:
- **Transfer Learning**: Utilizes pre-trained VGGish model for feature extraction
- **Automated Hyperparameter Optimization**: Uses Optuna for efficient hyperparameter search
- **Audio Preprocessing**: Comprehensive audio preprocessing pipeline including resampling, mel-spectrogram conversion, and normalization
- **Multi-species Classification**: Classifies 9 different marine mammal species
- **GPU Support**: Optimized for CUDA-enabled training

## Requirements

### Python Version
- Python 3.7 or higher

### Core Dependencies
```
torch>=1.9.0
torchaudio>=0.9.0
torchvggish
optuna>=2.10.0
pandas>=1.3.0
scikit-learn>=0.24.0
numpy>=1.21.0
tqdm>=4.62.0
openpyxl>=3.0.0
```

### Optional Dependencies (for extended preprocessing)
```
librosa>=0.8.0
noisereduce>=2.0.0
audiomentations>=1.0.0
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, recommended 16GB or more
- **Storage**: Sufficient space for audio datasets and model checkpoints

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd marine-mammal-classification
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install torch torchaudio torchvggish optuna pandas scikit-learn numpy tqdm openpyxl
```

4. **Download VGGish pre-trained weights:**
The code automatically downloads VGGish weights to:
```
~/.cache/torch/hub/checkpoints/vggish-10086976.pth
```

## Dataset Structure

The project expects the following directory structure:

```
project_root/
├── train/
│   ├── audio_file1.wav
│   ├── audio_file2.wav
│   └── ...
├── test/
│   ├── audio_file1.wav
│   ├── audio_file2.wav
│   └── ...
├── validate/
│   ├── audio_file1.wav
│   ├── audio_file2.wav
│   └── ...
├── train_metadata.xlsx
├── test_metadata.xlsx
└── validate_metadata.xlsx
```

### Metadata Format
Each Excel file should contain:
- **Column 0**: `file_name` - Audio file names
- **Column 1**: Additional metadata (optional)
- **Column 2**: `species_name` - Species labels for classification

## Code Architecture

### 1. MarineMammalDataset Class
Custom PyTorch Dataset class that handles:
- Audio file loading and preprocessing
- Mel-spectrogram conversion optimized for VGGish
- Label encoding for multi-class classification
- Audio normalization and padding/trimming

### 2. Model Architecture
- **Base Model**: Pre-trained VGGish for feature extraction
- **Custom Classifier**: Dynamically generated fully connected layers
- **Transfer Learning**: Feature extractor layers frozen, last 5 layers fine-tuned

### 3. Preprocessing Pipeline
- **Sample Rate**: 16kHz (VGGish requirement)
- **Duration**: 5 seconds per audio sample
- **Mel-Spectrogram**: 64 mel bins, 400 FFT window, 160 hop length
- **Output Shape**: 64x96 mel-spectrogram frames

## Hyperparameter Optimization

The Optuna optimization searches over the following hyperparameters:

| Parameter | Range | Type |
|-----------|-------|------|
| Number of Hidden Layers | 1-5 | Integer |
| Neurons per Layer | 512-8192 (step: 512) | Integer |
| Epochs | 4-12 (step: 2) | Integer |
| Learning Rate | 1e-5 to 1e-3 | Log-uniform |
| Dropout Rate | 0.1-0.5 (step: 0.1) | Float |
| Batch Size | [16, 32, 64] | Categorical |
| Optimizer | ['Adam', 'SGD', 'RMSprop'] | Categorical |
| Weight Decay | 1e-5 to 1e-3 | Log-uniform |

### Objective Function
The optimization maximizes test accuracy across 20 trials, with each trial involving:
1. Model architecture configuration
2. Training loop execution
3. Validation on test set
4. Return of test accuracy score

## Usage

### Basic Usage

1. **Update file paths** in the code to match your dataset location:
```python
train_annotation_file = "path/to/train_metadata.xlsx"
train_audio_dir = "path/to/train"
# ... update other paths
```

2. **Run hyperparameter optimization:**
```python
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

3. **Access best parameters:**
```python
print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)
```

### Advanced Usage

**Customize optimization parameters:**
```python
# Modify the objective function to adjust hyperparameter ranges
def objective(trial):
    # Custom parameter suggestions
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 2, 8)
    # ... other parameters
```

**Save and load studies:**
```python
# Save study
study.to_csv('optimization_results.csv')

# Load and continue optimization
study = optuna.load_study(study_name='study_name', storage='sqlite:///example.db')
```

## Results

Based on the optimization results shown in the notebook:

- **Best Test Accuracy**: 92.52%
- **Best Parameters**:
  - Hidden Layers: 1
  - Neurons per Layer: 4608
  - Epochs: 8
  - Learning Rate: 1.13e-05
  - Dropout Rate: 0.3
  - Batch Size: 64
  - Optimizer: RMSprop
  - Weight Decay: 1.60e-04

## Model Architecture

### VGGish Feature Extractor
- Pre-trained on AudioSet dataset
- Convolutional layers for audio feature extraction
- Outputs 12,288-dimensional feature vectors

### Custom Classifier
Dynamic fully connected network with:
- Configurable number of layers (1-5)
- Layer normalization
- ReLU activation
- Dropout regularization
- Final classification layer (9 classes)

### Training Strategy
- **Transfer Learning**: Freeze early VGGish layers, fine-tune last 5 layers
- **Loss Function**: CrossEntropyLoss
- **Optimization**: Adam/SGD/RMSprop with configurable parameters
- **Regularization**: Dropout and weight decay

## Performance Optimization

### GPU Utilization
- All tensors and models moved to GPU
- Efficient batch processing
- Memory-optimized data loading

### Data Loading
- Custom Dataset class with efficient audio loading
- Batched processing with configurable batch sizes
- Parallel data loading with DataLoader

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Reduce number of neurons per layer
   - Use gradient checkpointing

2. **Audio Loading Errors**:
   - Verify audio file formats (WAV recommended)
   - Check file paths in metadata
   - Ensure consistent sample rates

3. **Model Performance Issues**:
   - Increase number of optimization trials
   - Adjust hyperparameter ranges
   - Check data quality and preprocessing

### Debug Mode
Enable anomaly detection for gradient issues:
```python
torch.autograd.set_detect_anomaly(True)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- VGGish model from Google Research
- Optuna hyperparameter optimization framework
- PyTorch and torchaudio libraries
- Marine mammal research community

## Citation

If you use this code in your research, please cite:

```bibtex
@software{marine_mammal_classification,
  title={Marine Mammal Classification with VGGish and Optuna},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/marine-mammal-classification}
}
```