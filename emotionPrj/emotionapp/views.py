# from django.shortcuts import render
# from django.http import JsonResponse
# import torch
# import librosa 
# import numpy as np
# import pickle
# from django.http import HttpResponse

# Create your views here.



import os
import torch
import librosa
import numpy as np
import pickle
from django.http import JsonResponse
from django.shortcuts import render

from emotionapp.models import ParallelModel  # Ensure the model.py exists in the app

# Correct paths to your files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'emotionapp/static/Model/emotion_detection.pt')  # Update this path
SCALER_PATH = os.path.join(BASE_DIR, 'emotionapp/static/Model/scaler.pkl')  # Update this path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = ParallelModel(num_emotions=8, num_vocal_channels=2, num_intensity_levels=2, num_genders=2).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Load scaler
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading scaler: {e}")


def detect_emotion(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        try:
            # Read audio file from request
            audio_file = request.FILES['audio']
            y, sr = librosa.load(audio_file, sr=16000)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
            scaled_mfccs = scaler.transform(mfccs)

            # Convert features to tensor
            input_tensor = torch.tensor(scaled_mfccs, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict emotion
            with torch.no_grad():
                _, emotion_softmax, *_ = model(input_tensor)
                predictions = emotion_softmax.cpu().numpy().flatten()
                predicted_emotion = int(np.argmax(predictions))

            return JsonResponse({'emotion': predicted_emotion})

        except Exception as e:
            return JsonResponse({'error': f"Processing error: {e}"})
    return JsonResponse({'error': 'Invalid request'})




# Load model and scaler
# MODEL_PATH = 'path/to/emotion_detection.pt'
# SCALER_PATH = 'path/to/scaler.pkl'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print("DEBUG: Importing ParallelModel...")
# from .model import ParallelModel  
# Import the model class from the notebook (moved to `model.py`)
# print("DEBUG: Successfully imported ParallelModel.")
# model = ParallelModel(num_emotions=8, num_vocal_channels=2, num_intensity_levels=2, num_genders=2).to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()

# with open(SCALER_PATH, 'rb') as f:
#     scaler = pickle.load(f)

def index(request):
    context = {'title': 'Home'}
    return render(request, 'index.html',context)

# def detect_emotion(request):
#     if request.method == 'POST' and request.FILES['audio']:
#         audio_file = request.FILES['audio']
        
#         # Process audio
#         y, sr = librosa.load(audio_file, sr=16000)
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
#         scaled_mfccs = scaler.transform(mfccs)

#         # Convert to tensor
#         input_tensor = torch.tensor(scaled_mfccs, dtype=torch.float32).unsqueeze(0).to(device)

#         # Make prediction
#         with torch.no_grad():
#             _, emotion_softmax, *_ = model(input_tensor)
#             predictions = emotion_softmax.cpu().numpy().flatten()
#             predicted_emotion = np.argmax(predictions)

#         return JsonResponse({'emotion': predicted_emotion})
#     return JsonResponse({'error': 'Invalid request'})

def list(request):
    context = {'title': 'List'}
    return render(request, 'list.html', context)






