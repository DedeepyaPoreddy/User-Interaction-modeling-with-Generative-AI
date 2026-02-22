import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

plutchik_emotions = ['neutral', 'joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']

class T5WithEmotionEmbeddings(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.emotion_embedding = nn.Embedding(len(plutchik_emotions), config.d_model)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        emotion_ids=None,
        inputs_embeds=None,
        **kwargs
    ):
        # Handle cases where both input formats are None (during some generation steps)
        if input_ids is None and inputs_embeds is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        
        # Normal emotion-aware processing
        device = self.device
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        
        # Default to neutral if no emotion provided
        if emotion_ids is None:
            emotion_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)
        
        # Add emotion information
        emotion_embeds = self.emotion_embedding(emotion_ids).unsqueeze(1)
        inputs_embeds = inputs_embeds + emotion_embeds.expand(-1, inputs_embeds.size(1), -1)
        
        return super().forward(
            input_ids=None,  
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

config = T5Config.from_pretrained("./t5-emotion-embed-finetuned/model")

model = T5WithEmotionEmbeddings(config)

# print(model.emotion_embedding.weight.requires_grad) 
print(model.emotion_embedding.weight.grad)
