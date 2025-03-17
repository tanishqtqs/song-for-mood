import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.config import emotion_model_name, mood_map

emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)

def detect_emotion(sentence: str) -> str:
    inputs = emotion_tokenizer(sentence, return_tensors="pt")
    outputs = emotion_model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()    
    mood_keyword = mood_map.get(predicted_label, "neutral")
    return mood_keyword
