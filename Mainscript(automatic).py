from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import random
from openai import OpenAI
import pyaudio
import wave
import webrtcvad
import deepspeech
import pandas as pd

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Load the pre-trained BERT model and tokenizer.
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Prepare the input text by encoding it into a format suitable for the BERT model.
input_text = "This is an example sentence for processing."
encoded_input = tokenizer.encode_plus(
    input_text, 
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    return_tensors='pt'      # Return PyTorch tensors
)

# Pass the processed input through the BERT model. Captures contextual information about the input text.
outputs = model(**encoded_input)
last_hidden_state = outputs.last_hidden_state #contains the embeddings for each token in the input. Used for extracting features from specific tokens or phrases.
pooled_output = outputs.pooler_output #summary of the context of the whole input. Useful for understanding the overall sentiment.

# Defines a custom classifier for binary sentiment analysis.
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)


# Defines a custom Dataset for sentiment analysis.
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_pair = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoded_pair['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoded_pair['attention_mask'].squeeze(0)
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label)}


# Load dataset
df = pd.read_csv('/Users/archiebond/Documents/DE4/Masters/Code_tests/MovieReviewSmall.csv')
print('dataset read')
df['sentiment'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})
texts = df['review'].tolist() # Assuming text data is under the 'review' column
print(texts)
labels = df['sentiment'].tolist() # Assuming sentiment labels are in 'sentiment'
print(labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SentimentDataset(texts, labels, tokenizer, max_length=64)

training_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Set the number of training epochs (10 takes way too long)
number_of_epochs = 1  

# Example training loop (simplified)
classifier = SentimentClassifier()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)

# Assuming classifier and optimizer are defined, and the training loop is set up
print(f"Starting training for {number_of_epochs} epochs...")
for epoch in range(number_of_epochs):
    print(f"Epoch {epoch+1}/{number_of_epochs}")
    for batch in training_dataloader:
        # Accessing data directly from the dictionary keys
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        optimizer.zero_grad()
        # Make sure your classifier's forward method accepts all needed arguments
        outputs = classifier(input_ids, attention_mask)
        # Assuming outputs are logits, adjust if your classifier structure is different
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()




def predict_sentiment(text, model, tokenizer):
    # Tokenize and prepare input as done during training
    encoded_input = tokenizer.encode_plus(
        text, 
        add_special_tokens=True,
        max_length=64,  # Make sure this is the same as used in training
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move the tensor to the same device as the model
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    
    # Put model in evaluation mode
    model.eval()
    
    # No gradient needed for prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Process model output, e.g., apply softmax for probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    # Choose the predicted class (0 or 1) based on higher probability
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class

# Audio capture
def capture_audio():
    # Load the DeepSpeech model
    model_path = '/Users/archiebond/Documents/DE4/Masters/Code_tests/deepspeech-0.9.3-models.pbmm'
    scorer_path = '/Users/archiebond/Documents/DE4/Masters/Code_tests/deepspeech-0.9.3-models.scorer'
    model = deepspeech.Model(model_path)
    model.enableExternalScorer(scorer_path)

    # Set up audio recording
    vad = webrtcvad.Vad(1)  # Aggressiveness level
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  frames_per_buffer=320)

    print("Listening...")

    frames = []
    active = False
    num_silent_frames = 0

    try:
        while True:
            frame = stream.read(320, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, 16000)

            if active:
                frames.append(frame)
                if not is_speech:
                    num_silent_frames += 1
                    if num_silent_frames > 30:  # 0.6 seconds of silence
                        break
                else:
                    num_silent_frames = 0
            elif is_speech:
                active = True
                frames.append(frame)
                num_silent_frames = 0

        print("Processing speech...")

    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

    # Convert the list of frames to a byte array.
    audio_buffer = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Perform speech-to-text
    text = model.stt(audio_buffer)
    return text


# Function to generate an image based on text
def generate_image(text):
    OpenAI.api_key = 'XXX' #removed for privacy
    client = OpenAI()

    response = client.images.generate(
      model="dall-e-3",  # Specify the model, e.g., dall-e or dall-e-2
      prompt=text,
      n=1,  # Number of images to generate
      quality="standard",
      size="1024x1024"  # Specify the resolution of the image
    )
    return response


# Using the trained classifier model
if __name__ == "__main__":

    live_text = capture_audio()
    sentiment = predict_sentiment(live_text, classifier, tokenizer)
    print(f"Live Text: {live_text}\nPredicted Sentiment: {'Positive' if sentiment == 1 else 'Negative'}\n")

    if sentiment == 0:  # Assuming '0' is Negative
        print("Generating image for negative sentiment text...")
        image_response = generate_image(live_text)
        print(image_response)  # This will print the API response



