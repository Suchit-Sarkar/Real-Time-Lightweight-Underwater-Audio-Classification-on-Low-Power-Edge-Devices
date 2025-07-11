# Marine Mammal Classification with Knowledge Distillation

This project implements a marine mammal species classification system using knowledge distillation from a VGGish teacher model to a lightweight student model with attention mechanisms.

## Overview

The system classifies marine mammal vocalizations using a two-stage approach:
1. **Teacher Model**: Fine-tuned VGGish model for high-accuracy classification
2. **Student Model**: Lightweight CNN with attention mechanisms for efficient deployment

## Requirements

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tqdm
pip install librosa soundfile
pip install torchsummary
pip install noisereduce
pip install torchvggish
pip install scipy
pip install openpyxl  # For Excel file handling
```

### Hardware Requirements
- GPU with CUDA support (recommended)
- Minimum 8GB RAM
- Storage space for audio datasets

## Dataset Structure

The code expects the following directory structure:
```
project_root/
├── train/
│   └── audio_files.wav
├── test/
│   └── audio_files.wav
├── validate/
│   └── audio_files.wav
├── train_metadata.xlsx
├── test_metadata.xlsx
└── validate_metadata.xlsx
```

### Metadata Format
Excel files should contain columns:
- `file_name`: Audio file name
- `species_name`: Target species label
- `sample_rate`: Audio sample rate
- Feature columns: `spectral_centroid`, `rms_energy`, `zero_crossing_rate`, `spectral_rolloff`, `band_energy_ratio`, `mfcc_1` through `mfcc_13`

## Model Architecture

### Teacher Model (VGGish-based)
- **Input**: Mel spectrograms (96x64) + 18 additional features
- **Features**: VGGish backbone (12,288 features) + handcrafted audio features
- **Classifier**: 4-layer fully connected network with LayerNorm and dropout
- **Output**: 20 classes (19 species + 1 unknown)

### Student Model with Attention
- **CNN Backbone**: Lightweight 3-layer CNN (1→16→32→64 channels)
- **Attention**: CBAM (Convolutional Block Attention Module)
  - Channel attention using average and max pooling
  - Spatial attention using 7x7 convolution
- **Feature Fusion**: Combines CNN features with processed additional features
- **Classifier**: 2-layer fully connected network with LayerNorm

## Key Features

### Audio Processing Pipeline
1. **Preprocessing**: Mix-down, cut/pad to 5 seconds, float32 conversion
2. **Filtering**: Bandpass filter (3-19 kHz) for marine mammal frequency range
3. **Feature Extraction**: 
   - Mel spectrogram (64 mel bins, 400 FFT, 160 hop length)
   - 18 handcrafted features (spectral, temporal, MFCC)
4. **Normalization**: Custom normalization factors for each feature type

### Knowledge Distillation
- **Temperature**: 3.0 for soft target smoothing
- **Loss Function**: Weighted combination of soft (70%) and hard (30%) targets
- **Training**: 60 epochs with learning rate scheduling and early stopping

## Performance Optimizations

### Hyperparameter Tuning
- **Learning Rate**: AdamW optimizer with warmup (3 epochs) and ReduceLROnPlateau scheduling
- **Batch Size**: 16 (balanced for memory and convergence)
- **Dropout**: 0.3 and 0.2 for different layers
- **Weight Decay**: 0.01 for regularization
- **Gradient Clipping**: Max norm 1.0 to prevent exploding gradients

### Efficiency Improvements
- **Model Size**: ~90% reduction from teacher to student
- **Inference Speed**: Significant speedup with attention-based student
- **Memory Usage**: Reduced through efficient tensor operations

## Usage

### Training Knowledge Distillation
```python
# Load pretrained teacher model
teacher_model, config = load_model_for_inference(
    "path/to/teacher_model.pth", device
)

# Initialize student model
student = AudioTransformerStudent(num_additional_features=18, num_classes=20)

# Train with distillation
for epoch in range(num_distill_epochs):
    train_loss, train_acc = distill_train(
        student, teacher_model, train_loader, 
        optimizer, distill_criterion, device
    )
```

### Inference
```python
# Load trained student model
student.load_state_dict(torch.load("best_student_model.pth"))

# Process audio file
mel_spec, additional_features = process_audio_file("audio_file.wav")

# Make prediction
with torch.no_grad():
    output = student(mel_spec, additional_features)
    prediction = torch.argmax(output, dim=1)
```

## File Structure

### Key Files
- `only_dolphin_knowledge_distillation_attention_mechanism.py`: Main implementation
- `train_metadata.xlsx`: Training set metadata with precomputed features
- `test_metadata.xlsx`: Test set metadata
- `validate_metadata.xlsx`: Validation set metadata

### Output Files
- `best_student_with_attention_model.pth`: Best performing student model
- `final_student_model_with_attention.pth`: Final model with metadata
- `student_with_attention_misclassified_samples.xlsx`: Error analysis
- `train_vs_val_student_with_attention.xlsx`: Training curves

## Model Evaluation

### Metrics
- **Accuracy**: Classification accuracy with confidence threshold (0.7)
- **F1 Score**: Weighted F1 score for imbalanced classes
- **Confusion Matrix**: Detailed classification analysis
- **Confidence Threshold**: Unknown class assignment for low-confidence predictions

### Benchmarking
- **Feature Extraction Time**: ~0.1s per 5-second audio clip
- **Inference Time**: Sub-second prediction with student model
- **Model Size**: Compressed from VGGish to lightweight student

## Attention Mechanism Details

### CBAM (Convolutional Block Attention Module)
1. **Channel Attention**: 
   - Uses both average and max pooling
   - Shared MLP with 8:1 compression ratio
   - Sigmoid activation for attention weights

2. **Spatial Attention**:
   - Combines channel-wise average and max features
   - 7x7 convolution for spatial relationships
   - Sigmoid activation for pixel-wise attention

## Configuration

### Audio Parameters
- **Sample Rate**: 16 kHz (VGGish requirement)
- **Duration**: 5 seconds per sample
- **Mel Bins**: 64
- **FFT Size**: 400
- **Hop Length**: 160

### Training Parameters
- **Epochs**: 60 with early stopping
- **Batch Size**: 16
- **Learning Rate**: 0.001 with warmup and scheduling
- **Temperature**: 3.0 for distillation
- **Alpha**: 0.7 (soft target weight)

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Audio Loading Errors**: Check file paths and formats
3. **Feature Extraction Failures**: Verify audio file integrity
4. **Model Loading Issues**: Ensure consistent architecture definitions

### Performance Tips
- Use GPU for training and inference
- Precompute features when possible
- Monitor memory usage during training
- Use mixed precision for larger models

## Future Improvements

1. **Model Compression**: Quantization and pruning
2. **Real-time Processing**: Streaming audio support
3. **Multi-modal Learning**: Incorporate environmental data
4. **Transfer Learning**: Adaptation to new species
5. **Ensemble Methods**: Multiple student models

## Citation

If you use this code for research, please cite:
```bibtex
@article{marine_mammal_classification,
  title={Marine Mammal Classification with Knowledge Distillation and Attention},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact: [your.email@example.com]