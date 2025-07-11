# Underwater Audio Classification System

## Overview

This project is an underwater audio classification system that captures and analyzes underwater sounds in real-time using a Raspberry Pi. The system classifies various underwater sounds (like marine life, boats, etc.) and provides a web interface to visualize the results, including spectrograms and classification history.

## Features

* Real-time audio recording and processing
* Machine learning-based sound classification
* Web-based dashboard with:
  * Current detection results
  * Confidence levels
  * Spectrogram visualization
  * Detection history
* Adjustable update interval
* WebSocket-based real-time updates

## Requirements

### Hardware

* Raspberry Pi (any modern version - Pi 3B+ or Pi 4 recommended)
* Underwater microphone (hydrophone) with compatible audio interface
* MicroSD card (32GB or larger recommended)
* Internet connection (for remote access)

### Software

* Python 3.7+
* FastAPI
* Uvicorn
* PyTorch
* Librosa
* PyAudio
* NumPy
* SciPy
* scikit-learn
* Torchaudio

### Web Interface

* Modern web browser (Chrome, Firefox, Edge)
* WebSocket support

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/underwater-audio-classification.git
   cd underwater-audio-classification
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model:**
   ```bash
   # Download the model file and place it in the models/ directory
   wget https://your-model-hosting-url/underwater_audio_model.pth -O models/underwater_audio_model.pth
   ```

## Usage

### Starting the Backend

Run the FastAPI server:
```bash
python raspi_backend.py
```

The server will start on `http://0.0.0.0:8000`.

### Accessing the Web Interface

Open `index.html` in a web browser or access it through the server if hosted.

### Controls

1. **Connect**: Establish connection to the Raspberry Pi server
2. **Start Recording**: Begin capturing audio
3. **Stop Recording**: Stop audio capture
4. **Set Interval**: Adjust how often classifications are updated (0.5-10 seconds)

## Code Structure

### `raspi_backend.py`
FastAPI web server that handles:
* WebSocket connections for real-time updates
* REST endpoints for control (start/stop recording, set interval)
* Audio processing coordination

### `raspi_inference.py`
Contains the core audio processing and classification logic:
* Audio capture via PyAudio
* Feature extraction (mel spectrograms, MFCCs)
* Machine learning model (AudioTransformerStudent with CBAM attention)
* Inference pipeline

### `index.html`
Web interface with:
* Connection management
* Real-time visualization
* Interactive controls
* Responsive design for mobile/desktop

## Key Components

### Audio Processing
1. Captures audio at 16kHz sample rate
2. Applies bandpass filter (3kHz-19kHz)
3. Extracts mel spectrograms and additional features
4. Processes through neural network

### Machine Learning Model
* Custom CNN architecture with CBAM attention
* Trained to classify 20 underwater sound categories including:
  * Marine mammals (whales, dolphins)
  * Fish sounds
  * Boat engines
  * Underwater machinery
  * Ambient ocean sounds
* Includes confidence thresholding

### Web Interface
* Real-time updates via WebSocket
* Interactive spectrogram visualization
* History tracking of classifications
* Responsive design for all screen sizes

## Configuration

### Audio Settings
Edit the configuration in `raspi_inference.py`:
```python
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
```

### Model Settings
Adjust model parameters in `raspi_inference.py`:
```python
CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = 'models/underwater_audio_model.pth'
```

### Web Interface Settings
Modify connection settings in `index.html`:
```javascript
const WS_URL = 'ws://your-raspberry-pi-ip:8000/ws';
```

## Troubleshooting

### No Audio Input
* Verify hydrophone is properly connected
* Check PyAudio device configuration:
  ```bash
  python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"
  ```

### Connection Issues
* Ensure Raspberry Pi is on the same network
* Verify firewall allows WebSocket connections (port 8000)
* Check if the service is running:
  ```bash
  sudo netstat -tlnp | grep :8000
  ```

### Classification Errors
* Check model file is in correct location
* Verify audio preprocessing matches training configuration
* Monitor system resources (CPU/Memory usage)

### Performance Issues
* Reduce update interval for better performance
* Consider using a faster Raspberry Pi model
* Optimize audio buffer sizes

## File Structure

```
underwater-audio-classification/
├── README.md
├── requirements.txt
├── raspi_backend.py
├── raspi_inference.py
├── index.html
├── models/
│   └── underwater_audio_model.pth
├── static/
│   ├── css/
│   └── js/
└── data/
    └── audio_samples/
```

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
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* **Libraries and Frameworks:**
  * FastAPI for the web framework
  * PyTorch for machine learning capabilities
  * Librosa for audio processing
  * PyAudio for real-time audio capture
  
* **Research and Inspiration:**
  * Marine bioacoustics research community
  * Underwater sound classification papers
  * Open-source audio processing tools

* **Hardware Support:**
  * Raspberry Pi Foundation
  * Hydrophone manufacturers and marine audio equipment suppliers

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{underwater_audio_classification,
  title={Underwater Audio Classification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/underwater-audio-classification}
}
```

## Support

For questions, issues, or contributions, please:
* Open an issue on GitHub
* Contact the maintainers at [your-email@example.com]
* Join our discussion forum at [forum-link]

---

This system provides a complete pipeline for underwater sound monitoring, from audio capture to web-based visualization, making it suitable for marine research and environmental monitoring applications.