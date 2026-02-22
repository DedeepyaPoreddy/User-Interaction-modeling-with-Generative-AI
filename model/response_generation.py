import torch
from transformers import T5Config, AutoTokenizer, T5TokenizerFast, T5Tokenizer, AutoModelForSequenceClassification
from emotion_embed_T5_class import T5WithEmotionEmbeddings 
from emotion_classifier import classify_emotion
from bert_score import score
import warnings
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_embed = T5Tokenizer.from_pretrained(
    "./t5-emotion-embed-finetuned/tokenizer",
    add_prefix_space=False  
)

config_embed = T5Config.from_pretrained("./t5-emotion-embed-finetuned/model")

model_embed = T5WithEmotionEmbeddings.from_pretrained(
    "./t5-emotion-embed-finetuned/model",
    config=config_embed
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_embed.to(device)

plutchik_emotion2id = {
    'trust': 0,
    'joy': 1,
    'anger': 2,
    'anticipation': 3,
    'fear': 4,
    'sadness': 5,
    'disgust': 6,
    'surprise': 7,
    'neutral': 8
}

import random
import torch
from sentence_transformers import SentenceTransformer, util

# Set a random seed for reproducibility
torch.manual_seed(42)

# Load sentence transformer for semantic similarity
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Emotion Prompts Dictionary
emotion_prompts = {
    "sadness": [
        "I'm so sorry you're feeling this way. Want to talk about it?",
        "That sounds tough. I'm here if you need to share.",
        "I know it can feel overwhelming, but you're not alone.",
        "Sending you some warmth and comfort today."
    ],
    "joy": [
        "That's awesome! What's making you so happy?",
        "Yay! That sounds exciting!",
        "I'm so glad to hear you're feeling this way.",
        "Happiness looks great on you!"
    ],
    "fear": [
        "It’s okay to be scared. You’re not alone.",
        "Take a deep breath. You’ve got this.",
        "Fear is natural, but you're stronger than you think.",
        "I'm here for you. It will be okay."
    ],
    "anger": [
        "I can feel your frustration. Want to share what happened?",
        "It’s okay to be angry. It’s important to express it.",
        "That sounds so frustrating. I’m here to listen.",
        "Let it out. I’m all ears."
    ],
    "trust": [
        "It means a lot that you trust me.",
        "Thank you for sharing that. You can always count on me.",
        "Trust is so powerful. I’m honored you feel that way.",
        "You're not alone. I'll support you every step of the way."
    ],
    "anticipation": [
        "Sounds exciting! What are you looking forward to?",
        "I bet you can't wait! What’s coming up?",
        "I’m excited for you! Can’t wait to hear how it goes.",
        "Hope everything goes just the way you imagine!"
    ],
    "disgust": [
        "That sounds awful. I completely understand why you'd feel that way.",
        "Ugh, that must’ve been really unpleasant.",
        "Yuck! I can see why you're feeling disgusted.",
        "I'm sorry you had to experience that."
    ],
    "surprise": [
        "Wow! That’s unexpected! Tell me more!",
        "I can’t believe that! How did it happen?",
        "That’s such a twist! I’m curious, what happened next?",
        "Surprises sure make life interesting! What’s going on?"
    ],
    "neutral": [
        "I’m listening. Tell me more.",
        "I’m here for you.",
        "What’s been going on lately?",
        "I’m interested in what you have to say."
    ]
}

def generate_response(input_text):
    # Classify and get emotion
    emotion = classify_emotion(input_text)
    emotion_id = torch.tensor([plutchik_emotion2id.get(emotion, 8)], device=device)  # Default to 'neutral' if unknown
    print(f"\nDetected Emotion: {emotion} (ID: {emotion_id.item()})")

    # Select emotion prompt
    prompt_list = emotion_prompts.get(emotion, emotion_prompts["neutral"])
    selected_prompt = random.choice(prompt_list)

    # Combine selected prompt with the input text
    prompt_input = f"{selected_prompt} {input_text}"

    # Log the prompt being used
    # print(f"[Selected Prompt]: {selected_prompt}")
    # print(f"[Prompted Input]: {prompt_input}\n")

    # Tokenize input
    inputs = tokenizer_embed(prompt_input, return_tensors="pt", truncation=True, padding=True).to(device)

    # Generate multiple responses (5 candidates)
    try:
        outputs = model_embed.generate(
            **inputs,
            emotion_ids=emotion_id,
            max_length=50,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=5
        )
    except TypeError:
        # Fallback if emotion_ids not supported
        outputs = model_embed.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=5
        )

    # Decode generated responses
    responses = [tokenizer_embed.decode(o, skip_special_tokens=True) for o in outputs]

    # Calculate semantic similarity between input and responses
    input_emb = semantic_model.encode(input_text, convert_to_tensor=True)
    response_embs = semantic_model.encode(responses, convert_to_tensor=True)
    similarities = util.cos_sim(input_emb, response_embs)[0]  # shape: (5,)

    # Select best response based on similarity score
    best_idx = torch.argmax(similarities).item()
    best_response = responses[best_idx]

    print("\nGenerated Responses (Ranked by Similarity):")
    for i, (resp, score) in enumerate(zip(responses, similarities.tolist()), 1):
        print(f"{i}. ({score:.3f}) {resp}")

    return best_response


input_text = "I am excited about watching a movie tomorrow."
print("Input text:", input_text)
print(generate_response(input_text))

