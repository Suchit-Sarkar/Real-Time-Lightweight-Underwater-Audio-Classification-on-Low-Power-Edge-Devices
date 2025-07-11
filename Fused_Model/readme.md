The Dolphin_Updated_VGGish_hybrid_classifier_with_additional_features_with_noise.py file contains the training, validation and testing of the fused architecture.

Currently There are 19 different species of dolphins. More species Can be included given
the metadata files are updated with metadata creation pipeline.


System Requirements

Python 3.7+
CUDA-compatible GPU (recommended)
Windows/Linux/Mac OS

Required Python Packages
Core Dependencies
bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn tqdm matplotlib seaborn
pip install librosa soundfile scipy
pip install torchvggish torchsummary
pip install noisereduce
pip install openpyxl  # For Excel file handling
Alternative Installation (if pip fails)
bashconda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas numpy scikit-learn tqdm matplotlib seaborn librosa
pip install torchvggish torchsummary noisereduce soundfile openpyxl
Data Structure Required

Excel metadata files with columns: file_name, species_name, sample_rate, spectral_centroid, rms_energy, zero_crossing_rate, spectral_rolloff, band_energy_ratio, mfcc_1 through mfcc_13
Audio files in WAV format
Separate train/test/validate directories

Hardware Recommendations

GPU: 8GB+ VRAM
RAM: 16GB+ system memory
Storage: SSD recommended for faster data loading

Note
Update file paths in the script before running. The model requires VGGish pretrained weights which will be automatically downloaded to ~/.cache/torch/hub/checkpoints/
