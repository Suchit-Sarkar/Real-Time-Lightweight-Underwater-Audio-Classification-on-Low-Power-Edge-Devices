# Marine Mammal Audio Classification System

## Overview

This project is a comprehensive marine mammal audio classification system that captures and analyzes underwater sounds in real-time using a Raspberry Pi. The system focuses on dolphin vocalization classification across 19 species and provides a complete pipeline from data processing to real-time deployment with a web interface for visualization.

## 🚀 Quick Start

**Please Read This First:**
1. Download the vocalization of 19 species of dolphins from the Watkins marine mammal dataset
2. Apply data augmentation to generate your own test dataset
3. Use the PowerPoint presentation and report to reproduce the results
4. Download the pretrained VGGish model from Kaggle for the fused model results

## Features

* **Real-time audio recording and processing** on Raspberry Pi
* **Machine learning-based sound classification** for 19 dolphin species
* **Advanced model architectures** including fused models and knowledge distillation
* **Comprehensive data augmentation** pipeline
* **Hyperparameter optimization** for model performance
* **Web-based dashboard** with:
  * Current detection results
  * Confidence levels
  * Spectrogram visualization
  * Detection history
* **WebSocket-based real-time updates**
* **Full-stack implementation** with prototype deployment

## Project Structure

```
Real-Time-Lightweight-Underwater-Audio-Classification-on-Low-Power-Edge-Devices/
├── README.md
├── fused_model/
│   ├── readme.md
│   ├── fine_tuned_marine_vggish_combined_with_noise.pth
│   ├── initial_9_species_results/
│   └── 19_dolphin_species_results/
├── optuna_hyperparameter_optimization/
│   ├── readme.md
│   └── optimization_results/
├── knowledge_distillation/
│   ├── readme.txt
│   └── distillation
├── precomputation_n_metadata_creation/
|   ├── readme.md
├── data_augmentation/
│   ├── readme.md
│   └── data_augmentation.py
├── end_to_end_implementation/
│   ├── readme.md
│   ├── raspi_backend.py
│   ├── raspi_inference.py
│   ├── index.html
│   └── models/
└── docs/
    ├── presentation.pptx
    └── project_report.pdf
```

## Directory Contents

### 1. `fused_model_and_results/`
Contains the fused model architecture, implementation, saved models, and results:
- **Fused model architecture** combining VGGish with custom layers
- **Saved model**: `fine_tuned_marine_vggish_combined_with_noise.pth`
- **Initial 9 species results** and **19 dolphin species results**
- Complete implementation and evaluation metrics

### 2. `hyperparameter_optimization_and_results/`
Contains hyperparameter optimization implementation and results:
- Automated hyperparameter tuning scripts
- Grid search and random search implementations
- Performance comparison across different parameter combinations
- Optimization results and best parameter configurations

### 3. `knowledge_distillation_and_results/`
Contains knowledge distillation implementation and results:
- Teacher-student model architecture
- Distillation training pipeline
- Performance comparison between teacher and student models
- Compressed model for deployment

### 4. `data_augmentation/`
Contains data augmentation implementation and related files:
- Audio augmentation techniques (pitch shifting, time stretching, noise addition)
- Spectrogram augmentation methods
- Dataset expansion scripts
- Augmentation parameter configurations

### 5. `Full_stack_implementation/`
Contains the complete prototype implementation:
- Real-time audio processing system
- Web interface for visualization
- Raspberry Pi deployment scripts
- System integration components

## Requirements

### Hardware
* **Raspberry Pi** (Pi 3B+ or Pi 4 recommended)
* **Underwater microphone** (hydrophone) with compatible audio interface
* **MicroSD card** (32GB or larger recommended)
* **Internet connection** (for remote access)

### Software
* **Python 3.7+**
* **PyTorch 1.8+**
* **FastAPI**
* **Uvicorn**
* **Librosa**
* **PyAudio**
* **NumPy**
* **SciPy**
* **scikit-learn**
* **Torchaudio**
* **VGGish pretrained model** (download from Kaggle)

### Dataset
* **Watkins Marine Mammal Dataset** (19 dolphin species)
* **Total dataset size**: 13GB (not included in repository)
* **Contact for dataset access**: codephylic@gmail.com / suchitsarkar@yahoo.com

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/marine-mammal-audio-classification.git
   cd marine-mammal-audio-classification
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models:**
   ```bash
   # Download VGGish pretrained model from Kaggle
   # Place in fused_model_and_results/ directory
   
   # The fused model is already included:
   # fine_tuned_marine_vggish_combined_with_noise.pth
   ```

4. **Download the Watkins Marine Mammal Dataset:**
   - Contact: codephylic@gmail.com / suchitsarkar@yahoo.com
   - Apply data augmentation as described in the documentation

## Usage

### 1. Data Preparation
```bash
cd data_augmentation/
python augment_dataset.py --input_path /path/to/watkins/dataset --output_path ./augmented_data
```

### 2. Model Training (Optional)
```bash
cd fused_model_and_results/
python train_fused_model.py --config config.yaml
```

### 3. Hyperparameter Optimization
```bash
cd hyperparameter_optimization_and_results/
python optimize_hyperparameters.py --search_type grid --trials 100
```

### 4. Knowledge Distillation
```bash
cd knowledge_distillation_and_results/
python distill_model.py --teacher_model ../fused_model_and_results/fine_tuned_marine_vggish_combined_with_noise.pth
```

### 5. Full-Stack Deployment
```bash
cd Full_stack_implementation/
python raspi_backend.py
```

Access the web interface at `http://your-raspberry-pi-ip:8000`

## Model Architecture

### Fused Model
* **Base architecture**: VGGish pretrained model
* **Custom layers**: Additional classification layers for marine mammal sounds
* **Training**: Fine-tuned on 19 dolphin species from Watkins dataset
* **Augmentation**: Includes noise-robust training

### Knowledge Distillation
* **Teacher model**: Full fused model (`fine_tuned_marine_vggish_combined_with_noise.pth`)
* **Student model**: Compressed version for edge deployment
* **Performance**: Maintains accuracy while reducing computational requirements

## Key Components

### Audio Processing Pipeline
1. **Audio capture** at 16kHz sample rate
2. **Preprocessing** with bandpass filter (3kHz-19kHz)
3. **Feature extraction** (mel spectrograms, MFCCs)
4. **Model inference** through neural network
5. **Post-processing** and confidence thresholding

### Machine Learning Models
* **VGGish-based architecture** with custom classification head
* **19 dolphin species classification** capability
* **Confidence thresholding** for reliable predictions
* **Real-time inference** optimized for Raspberry Pi

### Web Interface
* **Real-time visualization** of audio spectrograms
* **Classification results** with confidence scores
* **Historical data tracking** and analysis
* **Responsive design** for mobile and desktop

## Performance Metrics

### Species Classification
* **19 dolphin species** from Watkins Marine Mammal Dataset
* **Initial 9 species results**: Available in `fused_model_and_results/`
* **Extended 19 species results**: Complete classification performance
* **Hyperparameter optimization**: Improved model performance
* **Knowledge distillation**: Compressed model with minimal accuracy loss

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   ```bash
   # Ensure VGGish pretrained model is downloaded
   # Check model file paths in configuration
   ```

2. **Audio Input Issues**:
   ```bash
   # Verify hydrophone connection
   python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"
   ```

3. **Memory Issues on Raspberry Pi**:
   ```bash
   # Use knowledge distilled model for better performance
   # Reduce batch size in inference
   ```

### Performance Optimization
* Use **knowledge distilled model** for edge deployment
* Implement **batch processing** for multiple audio segments
* Optimize **audio buffer sizes** for real-time processing

## Documentation

Each directory contains detailed instructions in `readme.txt` files:
- **fused_model_and_results/readme.txt**: Model architecture and training details
- **hyperparameter_optimization_and_results/readme.txt**: Optimization procedures
- **knowledge_distillation_and_results/readme.txt**: Distillation methodology
- **data_augmentation/readme.txt**: Augmentation techniques and parameters
- **Full_stack_implementation/readme.txt**: Deployment instructions

## Requirements.txt

```
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
torchaudio==2.1.0
librosa==0.10.1
pyaudio==0.2.11
numpy==1.24.3
scipy==1.11.3
scikit-learn==1.3.1
websockets==12.0
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3
tensorflow==2.13.0
```

## Reproduction Steps

1. **Download the Watkins Marine Mammal Dataset** (19 dolphin species)
2. **Apply data augmentation** using provided scripts
3. **Download VGGish pretrained model** from Kaggle
4. **Follow the PowerPoint presentation** and report for detailed methodology
5. **Use the fused model** `fine_tuned_marine_vggish_combined_with_noise.pth` for knowledge distillation
6. **Run each component** following the readme.txt instructions in each directory

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, dataset access, or collaborations:
- **Email**: codephylic@gmail.com
- **Alternative**: suchitsarkar@yahoo.com

## Acknowledgments

* **Watkins Marine Mammal Sound Database** for providing the dolphin vocalization dataset
* **VGGish model** from Google Research for audio feature extraction
* **Kaggle community** for pretrained model hosting
* **Marine bioacoustics research community** for foundational work
* **Open-source libraries**: PyTorch, LibROSA, FastAPI, and others

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{marine_mammal_audio_classification,
  title={Marine Mammal Audio Classification System: Real-time Dolphin Species Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/marine-mammal-audio-classification}
}
```

---

This comprehensive system provides an end-to-end solution for marine mammal audio classification, from advanced machine learning models to real-time deployment, making it suitable for marine research, environmental monitoring, and educational applications.