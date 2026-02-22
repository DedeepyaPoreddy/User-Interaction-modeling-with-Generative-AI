from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

#print("Loading Wav2Vec2 model and processor...")
processor_st = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model_st = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
#print("Wav2Vec2 model and processor loaded successfully!")