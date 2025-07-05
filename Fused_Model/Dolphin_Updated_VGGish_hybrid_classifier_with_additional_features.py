import os
import pandas as pd
import numpy as np
import torchaudio
import librosa
import soundfile as sf
import torchaudio.transforms as T
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
#from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter

# Split dataset into train, validation, and test sets
from torch.utils.data import random_split

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
    
if __name__ == "__main__":
    # Path to metadata and audio files
    
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
    # Initialize the VGGish model
    model = vggish().to(device)
    def new_forward(x,additional_features):
        with torch.set_grad_enabled(True):
            vggish_embeddings = model.features(x).view(x.size(0), -1)
            #x = model.features(x)  # Extract features
            #x = x.view(x.size(0), -1)  # Flatten the feature map
            if additional_features is not None:
                combined_features = torch.cat([vggish_embeddings, additional_features], dim=1)
            else:
                combined_features = vggish_embeddings
            x = model.fc(combined_features)  # Pass through the updated classifier
        return x
    #model=CustomVGGish().to(device)
    # Move PCA tensors to the same device as the model
    def create_fc(num_inputs,num_layers,neuron_per_layer,num_outputs,dropout_rate):
        layers=[]
        for i in range(num_layers):
            layers.append(nn.Linear(num_inputs,neuron_per_layer))
            layers.append(nn.LayerNorm(neuron_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate, inplace=False))
            num_inputs=neuron_per_layer
            neuron_per_layer=num_inputs//2
        layers.append(nn.Linear(num_inputs,num_outputs))
 #Create the sequential layer 
        fc=nn.Sequential(*layers)
        return fc
    num_hidden_layers=4 #change to 4
    neurons_per_layer=512
    model.fc=create_fc(12288+num_additional_features,num_hidden_layers, neurons_per_layer,19,0.2).to(device)

    model.forward=new_forward
    if model.postprocess:
        model.pproc._pca_matrix = model.pproc._pca_matrix.to(device)
        model.pproc._pca_means = model.pproc._pca_means.to(device)

    input_tensor = torch.rand(1, 1, 96, 64).to(device)
    #output = model(input_tensor)
    pretrained_weights_path = "C:/Users/myair/.cache/torch/hub/checkpoints/vggish-10086976.pth"

    # # Load pretrained weights
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device, weights_only=True), strict=False)
    # # Freeze pretrained feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False
    num_classes = 20
    confidence_threshold = 0.7 

    #output = model(input_tensor)
    #print(output.shape)
    #print(len(train_dataset.label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=(1e-5)*4)
    optimizer = optim.RMSprop(model.parameters(), lr=0.0002367502371325163, weight_decay=2.9241175186526862e-05)

    num_epochs = 12

    # adaptive learning Rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    print("new model #################\n",model)
    # Unfreeze the last two layers of the features part
    for param in list(model.features[-3:].parameters()):  # Accessing the last  layers
        param.requires_grad = True
   
    # try:
    # #from torchsummary import summary
    #     print("\n=== Detailed Model Summary ===")
    #     # Note: We need to provide input sizes for both mel_spec and additional_features
    #     # Since summary can't handle multiple inputs directly, we'll show them separately
    #     print("Mel Spectrogram path summary:")
    #     summary(model.features, (1, 96, 64), device=device)
    #     print("\nClassifier path summary:")
    #     summary(model.fc, (12288 + 18,), device=device)
    #     model.eval()
    # except ImportError:
    #     print("torchsummary not available, skipping detailed summary")

    # Define validation function
    def validate_model(model, val_loader, criterion):
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, additional_features, labels in val_loader:
                inputs,additional_features, labels = inputs.to(device),additional_features.to(device), labels.to(device)
                #print(additional_features)
                # Forward pass
                outputs = model(inputs, additional_features)
                loss = criterion(outputs, labels)

                # Predictions
                probs = torch.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, 1)
                preds[max_probs < confidence_threshold] = num_classes-1
                
                # Update metrics
                running_loss += loss.item() * inputs.size(0)               
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                # Collect predictions for F1 score
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        return epoch_loss, epoch_accuracy, epoch_f1


# Training loop with validation
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs,additional_features,labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, additional_features, labels = inputs.to(device),additional_features.to(device), labels.to(device)
            #print(f"Additional features: {len(additional_features)}\n")
            # Forward pass
            outputs = model(inputs,additional_features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        val_loss, val_accuracy, val_f1 = validate_model(model, val_loader, criterion)
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
    # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, "
          f"LR: {current_lr:.2e}")


    # Test the model
    # Updated test_model function with confusion matrix
    # Updated test_model function with misclassification logging
    def test_model(model, test_loader, label_encoder, output_file="E:/Project_Experiments/only_dolphin_experiment/misclassified_samples.xlsx"):
        model.eval()
        all_preds = []
        all_labels = []
        misclassified_samples = []
        global_index = 0
        with torch.no_grad():
            for inputs,additional_features, labels in test_loader:
                inputs,additional_features, labels = inputs.to(device), additional_features.to(device),labels.to(device)

            # Forward pass
                outputs = model(inputs, additional_features)
                probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
                max_probs, preds = torch.max(probs, dim=1)
                #print(probs.shape)

            # Classify as "unknown" if confidence is below the threshold
                preds[max_probs < confidence_threshold] = num_classes-1  # Assign to the unknown class

            # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Collect misclassified samples
                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        file_path = test_dataset._get_audio_sample_path(global_index+i)  # Get file path
                        actual_class = label_encoder.inverse_transform([labels[i].item()])[0]
                        classified_to = label_encoder.inverse_transform([preds[i].item()])[0]
                        misclassified_samples.append({
                            "filename": file_path,
                            "actual class": actual_class,
                            "classified to": classified_to
                        })
                global_index += len(labels)
    # Save misclassified samples to Excel
        if misclassified_samples:
            df = pd.DataFrame(misclassified_samples)
            df.to_excel(output_file, index=False)
            print(f"Misclassified samples saved to {output_file}")

    # Calculate the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("Classification Report:\n", classification_report(all_labels, all_preds))

    # Visualize the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

# Call the test_model function
    test_model(model, test_loader, train_dataset.label_encoder)


    # Save the fine-tuned model
    #torch.save(model, "E:/Project_Experiments/only_dolphin_experiment/fine_tuned_marine_vggish_combined.pth")
    # At the end of your script, replace the current model saving line with:

# Save the model state, label encoder, and other necessary information
model_save_path = "E:/Project_Experiments/only_dolphin_experiment/fine_tuned_marine_vggish_combined.pth"

# Create a dictionary with all necessary components
save_dict = {
    'model_state_dict': model.state_dict(),
    'label_encoder_classes': train_dataset.label_encoder.classes_,
    'num_additional_features': num_additional_features,
    'confidence_threshold': confidence_threshold,
    'feature_columns': train_dataset.feature_columns,
    'sample_rate': sample_rate,
    'num_samples': num_samples
}

# Save with weights_only=False since we need to save numpy arrays (label encoder classes)
torch.save(save_dict, model_save_path, _use_new_zipfile_serialization=True)
print(f"Model and necessary components saved to {model_save_path}")
