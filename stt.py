from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
import torch
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def transcribe_audio(audio_file_path):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = tokenizer(waveform.squeeze().numpy(), return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription
  
texts = [
    "example transcription in German",
    "example transcription in Turkish",
    "code switching example"
]
labels = [
    "German",
    "Turkish",
    "Code-Switched"
]

vectorizer = CountVectorizer()
classifier = MultinomialNB()
pipeline = make_pipeline(vectorizer, classifier)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
pipeline.fit(X_train, y_train)

def classify_text(text):
    return pipeline.predict([text])[0]

def classify_audio_files(root_directory):
    for subdir, _, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith(".wav"):
                file_path = os.path.join(subdir, filename)
                print(f"Processing file: {filename}")
                transcription = transcribe_audio(file_path)
                label = classify_text(transcription)
                print(f"File: {filename}, Transcription: {transcription}, Label: {label}")

# Update the directory path to point to the root directory containing all subdirectories
directory_path = "/Users/jasonthorn/SAGTAudioDatabasev1/audio"
classify_audio_files(directory_path)

