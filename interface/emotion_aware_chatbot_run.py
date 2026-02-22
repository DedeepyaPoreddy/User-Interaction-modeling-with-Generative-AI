from speech_model import processor_st, model_st
from response_generation import generate_response
from emotion_embed_T5_class import T5WithEmotionEmbeddings 
from emotion_classifier import classify_emotion
import os
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play
import speech_recognition as sr
import io
import warnings
import pyttsx3
import re
from tkinter import *
from tkinter import ttk, scrolledtext
from threading import Thread

warnings.filterwarnings("ignore")

class VoiceAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Aware Chatbot")
        self.root.geometry("600x400")
        self.engine = pyttsx3.init()
        self.setup_ui()
        self.is_recording = False

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="Emotion Aware Chatbot", font=('Helvetica', 14, 'bold')).pack(pady=10)

        # Record button
        self.record_btn = ttk.Button(main_frame, text="🎤 Start Recording (50s)", command=self.start_recording_thread)
        self.record_btn.pack(pady=10)

        # Status label
        self.status_var = StringVar()
        self.status_var.set("Ready to record")
        ttk.Label(main_frame, textvariable=self.status_var).pack()

        # Transcription display
        ttk.Label(main_frame, text="Transcription:").pack(anchor=W)
        self.transcription_text = scrolledtext.ScrolledText(main_frame, height=5, wrap=WORD)
        self.transcription_text.pack(fill=X, pady=5)

        # Response display
        ttk.Label(main_frame, text="Assistant Response:").pack(anchor=W)
        self.response_text = scrolledtext.ScrolledText(main_frame, height=5, wrap=WORD)
        self.response_text.pack(fill=X, pady=5)

        # Emotion display
        self.emotion_var = StringVar()
        ttk.Label(main_frame, textvariable=self.emotion_var, font=('Helvetica', 12)).pack(pady=5)

    def start_recording_thread(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.config(text="⏹ Recording...")
            Thread(target=self.record_and_process).start()

    def record_and_process(self):
        self.status_var.set("Recording... Speak now")
        audio_segment, wav_data = self.record_audio_from_mic()
        
        if audio_segment and wav_data:
            self.status_var.set("Processing...")
            self.process_audio_data(audio_segment, wav_data)
        
        self.is_recording = False
        self.record_btn.config(text="🎤 Start Recording (50s)")
        self.status_var.set("Ready to record")

    def speak_text(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def clean_transcription(self, text):
        text = re.sub(r'<[^>]+>', '', text)
        text = ' '.join(text.split())
        if text:
            text = text[0].upper() + text[1:]
        return text

    def audio_to_text(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700")
            model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700")
            
            input_values = processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            
            return self.clean_transcription(transcription)
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

    def record_audio_from_mic(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio_data = recognizer.listen(source, phrase_time_limit=50)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                audio_segment = AudioSegment.from_file(wav_data, format="wav")
                return audio_segment, wav_data
            except Exception as e:
                print(f"Recording error: {e}")
                return None, None

    def process_audio_data(self, audio_segment, wav_data):
        with open("temp_input.wav", "wb") as f:
            wav_data.seek(0)
            f.write(wav_data.read())
        
        transcription = self.audio_to_text("temp_input.wav")
        os.remove("temp_input.wav")
        
        if not transcription:
            self.status_var.set("Could not transcribe audio. Please try again.")
            return
        
        self.transcription_text.delete(1.0, END)
        self.transcription_text.insert(END, transcription)
        
        response = generate_response(transcription)
        
        self.response_text.delete(1.0, END)
        self.response_text.insert(END, response)
        
        # Detect emotion (assuming your classify_emotion function exists)
        emotion = classify_emotion(transcription)
        self.emotion_var.set(f"Detected Emotion: {emotion}")
        
        self.speak_text(response)

if __name__ == "__main__":
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    root = Tk()
    app = VoiceAssistantApp(root)
    root.mainloop()