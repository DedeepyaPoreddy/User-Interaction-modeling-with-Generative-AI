import torch
import pandas as pd
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

# def generate_response(input_text):
#     model_embed.to(device)

#     emotion = classify_emotion(input_text)
#     emotion_id = torch.tensor([plutchik_emotion2id.get(emotion, 8)], device=device)

#     inputs = tokenizer_embed(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

#     try:
#         outputs = model_embed.generate(
#             **inputs,
#             emotion_ids=emotion_id,
#             max_length=50,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.7
#         )
#     except TypeError:
#         outputs = model_embed.generate(
#             **inputs,
#             emotion_ids=emotion_id,
#             max_length=50,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.7,
#     num_return_sequences=5
#         )

#     responses = [tokenizer_embed.decode(o, skip_special_tokens=True) for o in outputs]

    
#     P, R, F1 = score(responses, [input_text]*len(responses), lang='en', verbose=False)
#     best_response = responses[F1.argmax()]

#     return best_response

# def generate_response(input_text):
#     # model_embed.to(device)

#     # Classify and print emotion
#     emotion = classify_emotion(input_text)
#     emotion_id = torch.tensor([plutchik_emotion2id.get(emotion, 8)], device=device)
#     print(f"\nDetected Emotion: {emotion} (ID: {emotion_id.item()})")  # <-- ADDED THIS LINE

#     inputs = tokenizer_embed(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

#     try:
#         outputs = model_embed.generate(
#             **inputs,
#             emotion_ids=emotion_id,
#             max_length=50,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.7
#         )
#     except TypeError:
#         outputs = model_embed.generate(
#             **inputs,
#             emotion_ids=emotion_id,
#             max_length=50,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.7,
#             num_return_sequences=5
#         )

#     responses = [tokenizer_embed.decode(o, skip_special_tokens=True) for o in outputs]
    
#     P, R, F1 = score(responses, [input_text]*len(responses), lang='en', verbose=False)
#     best_response = responses[F1.argmax()]

#     return best_response


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
    similarities = util.cos_sim(input_emb, response_embs)[0]  

    # Select best response based on similarity score
    best_idx = torch.argmax(similarities).item()
    best_response = responses[best_idx]

    print("\nGenerated Responses (Ranked by Similarity):")
    for i, (resp, score) in enumerate(zip(responses, similarities.tolist()), 1):
        print(f"{i}. ({score:.3f}) {resp}")

    return best_response




# def generate_response(input_text):
#     model_embed.to(device)

#     # Classify the emotion
#     emotion = classify_emotion(input_text)
#     emotion_id = torch.tensor([plutchik_emotion2id.get(emotion, 0)], device=device)  # Default to 'neutral' if unknown

#     # Tokenize input
#     inputs = tokenizer_embed(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

#     # Generate with emotion embedding
#     try:
#         outputs = model_embed.generate(
#             **inputs,
#             emotion_ids=emotion_id,
#             max_length=50,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.7
#         )
#         responses = [tokenizer_embed.decode(outputs[0], skip_special_tokens=True)]
#     except TypeError:
#         outputs = model_embed.generate(
#             **inputs,
#             emotion_ids=emotion_id,
#             max_length=50,
#             do_sample=False,
#             top_p=0.95,
#             temperature=0.7,
#             num_return_sequences=5
#         )
#         responses = [tokenizer_embed.decode(o, skip_special_tokens=True) for o in outputs]

#     P, R, F1 = score(responses, [input_text]*len(responses), lang='en', verbose=False)
#     best_response = responses[F1.argmax()]

#     return best_response

# import random
# import torch

# def generate_response(input_text):
#     model_embed.to(device)

#     # Classify the emotion
#     emotion = classify_emotion(input_text)
#     emotion_id = plutchik_emotion2id.get(emotion, 0)

#     # Define prompt templates for each emotion
#     emotion_prompts = {
#         "sadness": [
#             "You're not alone. Things will get better.",
#             "I’m here for you. It's okay to feel this way.",
#             "That sounds really hard. Sending you warmth.",
#             "You're strong, even when it doesn’t feel like it."
#         ],
#         "joy": [
#             "That's wonderful to hear!",
#             "So glad you're feeling this way!",
#             "Happiness looks great on you.",
#             "Keep that joy flowing!"
#         ],
#         "fear": [
#             "It’s okay to be scared. You’re not alone.",
#             "Take a deep breath. You’ve got this.",
#             "Fear is natural. You're safe here.",
#             "I hear you. It will be okay."
#         ],
#         "anger": [
#             "That sounds frustrating. I’m here for you.",
#             "Anger shows something matters to you.",
#             "You have every right to feel upset.",
#             "Let it out — I’m listening."
#         ],
#         "trust": [
#             "I'm glad you feel that way.",
#             "You can count on me.",
#             "That means a lot — thank you.",
#             "Trust makes everything better."
#         ],
#         "anticipation": [
#             "Sounds exciting!",
#             "I bet you’re looking forward to it.",
#             "Can’t wait to see how it turns out.",
#             "Hope it brings you joy!"
#         ],
#         "disgust": [
#             "That must’ve been awful.",
#             "Yuck! That sounds terrible.",
#             "I can see why that upset you.",
#             "I’d feel the same way."
#         ],
#         "surprise": [
#             "Wow, that’s great!",
#             "Oh! Tell me more.",
#             "That's quite a twist!",
#             "Surprises keep life interesting."
#         ],
#         "neutral": [
#             "I’m listening.",
#             "Go on, I’m here.",
#             "Tell me more about it.",
#             "I'm with you."
#         ]
#     }

#     # Select prompt
#     prompt_list = emotion_prompts.get(emotion, emotion_prompts["neutral"])
#     selected_prompt = random.choice(prompt_list)

#     # Combine prompt with input
#     prompt_input = f"{selected_prompt} {input_text}"

#     # # Log what is being sent to the model
#     # print(f"\n[Detected Emotion]: {emotion}")
#     # print(f"[Selected Prompt]: {selected_prompt}")
#     # print(f"[Prompted Input]: {prompt_input}\n")

#     # Tokenize input
#     inputs = tokenizer_embed(prompt_input, return_tensors="pt", truncation=True, padding=True).to(device)

#     # Generate response
#     try:
#         outputs = model_embed.generate(
#             **inputs,
#             max_length=50,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.7
#         )
#         responses = [tokenizer_embed.decode(outputs[0], skip_special_tokens=True)]
#     except TypeError:
#         outputs = model_embed.generate(
#             **inputs,
#             max_length=50,
#             do_sample=False,
#             top_p=0.95,
#             temperature=0.7,
#             num_return_sequences=5
#         )
#         responses = [tokenizer_embed.decode(o, skip_special_tokens=True) for o in outputs]

#     # Pick best using F1 score
#     P, R, F1 = score(responses, [input_text] * len(responses), lang='en', verbose=False)
#     best_response = responses[F1.argmax()]

#     # Combine the selected prompt with the model's best response
#     final_response = f"{best_response} {selected_prompt}"
#     return final_response


# input_text = "I am excited about watching a movie tomorrow."
# print("Input text:", input_text)
# print(generate_response(input_text))

test_df = pd.read_pickle('./dataset_emotions/test_emotions.pkl')

batch_size = 100
num_batches = int(len(test_df) / batch_size)

for batch in range(num_batches):
    start_index = batch * batch_size
    # start_index = batch * batch_size + 2600
    end_index = min(start_index + batch_size, len(test_df))
    batch_data = test_df[start_index:end_index]
    reference_sentences = []
    generated_sentences = []
    for index, row in batch_data.iterrows():
        input_text, label = row['conversation']['input_text'], row['conversation']['label']
        generated_sentences.append(generate_response(input_text))
        reference_sentences.append(label)
    df = pd.DataFrame({'reference_sentences': reference_sentences,
                       'generated_sentences': generated_sentences})
    df.to_pickle(f'./Generated_files/ref_gen_{start_index}.pkl')

import os
df_list = []

directory = './Generated_files/'

for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory, filename)
        df_list.append(pd.read_pickle(filepath))

merged_df = pd.concat(df_list, ignore_index=True)
merged_df.shape
merged_df.to_pickle('./Generated_files/ref_gen_master.pkl')

import evaluate

from bert_score import score

precision, recall, f1 = score(
    generated_sentences, reference_sentences, lang='en', verbose=False)

print(f"BERT score (F1): {f1.mean().item():.2f}")


