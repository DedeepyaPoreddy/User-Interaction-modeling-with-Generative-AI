from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize emotion classification model and tokenizer
emotion_model = AutoModelForSequenceClassification.from_pretrained(
    "joeddav/distilbert-base-uncased-go-emotions-student"
).to(device)

emotion_tokenizer = AutoTokenizer.from_pretrained(
    "joeddav/distilbert-base-uncased-go-emotions-student"
)

go_to_plutchik = {
    'admiration': 'trust', 'amusement': 'joy', 'anger': 'anger',
    'annoyance': 'anger', 'approval': 'joy', 'caring': 'trust',
    'confusion': 'anticipation', 'curiosity': 'anticipation',
    'desire': 'anticipation', 'disappointment': 'sadness',
    'disapproval': 'disgust', 'disgust': 'disgust', 'embarrassment': 'anticipation',
    'excitement': 'joy', 'fear': 'fear', 'gratitude': 'trust', 'grief': 'sadness',
    'joy': 'joy', 'love': 'trust', 'nervousness': 'fear', 'optimism': 'joy',
    'pride': 'joy', 'realization': 'surprise', 'relief': 'anticipation',
    'remorse': 'sadness', 'sadness': 'sadness', 'surprise': 'surprise'
}

def classify_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    top_class = torch.argmax(probs, dim=1).item()
    go_emotion = emotion_model.config.id2label[top_class]
    return go_to_plutchik.get(go_emotion, "neutral")

# print(classify_emotion("I am excited about tomorrow")) 