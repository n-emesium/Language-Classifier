import torchaudio
import torch
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53")

def classify_audio(audio_file_path):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    class_labels = ["German", "Turkish", "Code-Switched"]  
    return class_labels[predicted_class]

def classify_audio_files(root_directory):
    for subdir, _, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith(".wav"):
                file_path = os.path.join(subdir, filename)
                print(f"Processing file: {filename}")
                try:
                    label = classify_audio(file_path)
                    print(f"File: {filename}, Label: {label}")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

directory_path = "/Users/jasonthorn/SAGTAudioDatabasev1/audio"
classify_audio_files(directory_path)
