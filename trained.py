import os
import torchaudio
import torch
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=3)

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform

def load_dataset_from_directory(directory_path):
    audio_data = []
    labels = []
    class_labels = ["German", "Turkish", "Code-Switched"]
    for subdir, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".wav"):
                file_path = os.path.join(subdir, filename)
                print(f"Processing file: {filename}")
                try:
                    folder_name = os.path.basename(subdir).lower()
                    if folder_name.startswith("calla"):
                        label_index = class_labels.index("Code-Switched")
                    elif folder_name.startswith("efeu"):
                        label_index = class_labels.index("German")
                    elif folder_name.startswith("senna"):
                        label_index = class_labels.index("Turkish")
                    else:
                        raise ValueError(f"Unrecognized folder name: {folder_name}")
                    
                    audio_data.append({"path": file_path, "label": label_index})
                except ValueError as e:
                    print(f"Error processing file {filename}: {e}")
    
    return Dataset.from_dict({"path": [item['path'] for item in audio_data],
                              "label": [item['label'] for item in audio_data]})

directory_path = "/Users/jasonthorn/SAGTAudioDatabasev1/audio"
dataset = load_dataset_from_directory(directory_path)

def preprocess_data(batch):
    waveform = load_audio(batch['path'])
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    return batch

dataset = dataset.map(preprocess_data, remove_columns=["path"])
train_test_ratio = 0.2
train_dataset = dataset.train_test_split(test_size=train_test_ratio)['train']
test_dataset = dataset.train_test_split(test_size=train_test_ratio)['test']

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
    data_collator=lambda data: {
        'input_values': torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['input_values']) for f in data], batch_first=True, padding_value=0.0
        ),  
        'attention_mask': torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['attention_mask']) for f in data], batch_first=True, padding_value=0
        ),  
        'labels': torch.tensor([f['label'] for f in data])  
    },
)


#Old trainer, this doesn't work. 

"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
    data_collator=lambda data: {
        'input_values': torch.stack([f['input_values'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.tensor([f['label'] for f in data])
    },
)
"""
trainer.train()
results = trainer.evaluate()
print(results)

model.save_pretrained("./fine_tuned_model")
processor.save_pretrained("./fine_tuned_model")
