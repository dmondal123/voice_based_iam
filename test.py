import torch
import torchaudio
from transformers import Wav2Vec2Processor
import torch.nn as nn
import torch.nn.functional as F
import os
# CNN model definition
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 20000, 128)  # Adjusted for input size 80000
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 20000)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the trained model
def load_model(model_path, num_classes):
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Prediction function
def predict_user(audio_file, model, processor, target_sr=16000, max_length=16000*5):
    audio, sr = torchaudio.load(audio_file)
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    audio = audio.squeeze()[:max_length]
    if audio.shape[0] < max_length:
        padding = max_length - audio.shape[0]
        audio = torch.nn.functional.pad(audio, (0, padding))
    
    inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].squeeze(0)
    input_values = input_values.unsqueeze(0)
    input_values = input_values.reshape(input_values.shape[0], -1)

    with torch.no_grad():
        outputs = model(input_values)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
    
    user_dirs = [d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))]
    user_dirs.append('Unknown')
    predicted_user = user_dirs[predicted.item()]
    
    if predicted_user == 'Unknown' or probabilities[0][predicted].item() < 0.5:  # Example threshold
        return "User not in database"
    return predicted_user

model_path = "model_checkpoint.pth"  # Path to the saved model
audio_file = "/Users/dmondal/Documents/iam/testing/test/testing_anushka_001.wav"  # Path to the audio file for prediction
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
num_classes = len([d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))]) + 1  # Adding 1 for "Unknown"

# Load the model
model = load_model(model_path, num_classes)

# Predict the user
predicted_user = predict_user(audio_file, model, processor)
print(f"Predicted User: {predicted_user}")
