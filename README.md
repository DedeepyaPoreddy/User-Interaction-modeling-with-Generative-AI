# Emotion-Aware Human-Robot Interaction Using T5 Transformers

This project introduces a modified T5 Transformer model designed for emotionally intelligent dialogue. Unlike standard models that treat emotion as a simple text prefix, this architecture integrates trainable emotion embeddings directly into the model’s internal representations. This allows for more nuanced and empathetic responses in human-robot interaction (HRI) and mental health support [cite:1].

## Project Overview

- **Emotion-Aware Architecture**: Enhances the T5 model with a custom embedding layer mapped to Plutchik’s eight primary emotions: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, and Anticipation [cite:1].
- **Speech-to-Speech Loop**: Integrated voice-based interaction using Wav2Vec2 for speech-to-text and pyttsx3 for text-to-speech.
- **Preprocessing Pipeline**: A comprehensive data workflow for the EmpatheticDialogues dataset, including fine-grained emotion classification via DistilBERT.
- **Interactive GUI**: A user-friendly Tkinter interface that displays real-time transcription, detected emotion, and generated responses.

## Key Features

- **Fine-Grained Control**: Token-level emotion conditioning provides more expressive and contextually aligned outputs compared to baseline methods.
- **Dual Training**: The model is trained to simultaneously understand emotional intent and generate semantically relevant responses.
- **Speech Integration**: Supports hands-free interaction, making it suitable for social robotics and assistive technologies.
- **Semantic Filtering**: Uses SentenceTransformers to ensure generated responses maintain high semantic similarity to user inputs.

## Repository Structure

- `model/`: Contains the custom `T5WithEmotionEmbeddings` class and training logic.
- `data/`: Scripts for cleaning and labeling the EmpatheticDialogues dataset.
- `interface/`: The Tkinter-based GUI for speech-to-speech interaction.
- `evaluation/`: Scripts for calculating BLEU, ROUGE, and BERTScore metrics.

## Methodology

1.  **Data Preprocessing**: Segmenting dialogues and mapping emotions to Plutchik’s taxonomy.
2.  **Custom Class Creation**: Extending `T5ForConditionalGeneration` to include learned emotion vectors.
3.  **Model Fine-Tuning**: Training on emotion-tagged dialogue tuples using a custom Seq2Seq trainer.
4.  **Response Generation**: Orchestrating emotion recognition, prompting, and reranking for final output.

## Summary

By teaching AI to process feelings as part of its core logic, this project moves beyond simple text matching toward true digital empathy. The system is a step forward for social robots and digital mental health platforms, proving that AI can be built to understand not just what is said, but how it is felt.
