# Multi-Modal Emotion Recognition Project

## Overview
This project builds a lightweight **multi-modal emotion recognition** system based on facial and audio features.  
The full pipeline includes:
- Preprocessing emotion labels
- Cleaning invalid videos
- Loading and augmenting datasets
- Training and evaluating a custom-designed CNN+LSTM model for multi-modal emotion classification

## Project Structure
Project Structure

```pgsql
├── csv_convert.py         — Convert emotion labels from text to numeric format
├── clean_csv.py           — Remove entries with inaccessible videos
├── data/
│   ├── meld_dataloader.py — MELD dataset loader
│   └── collate_fn.py      — Custom collate function for batching
├── modules/
│   ├── mobilefacenet.py     — CNN model for face frames
│   ├── audio_model.py       — CNN model for audio spectrograms
│   ├── face_detect.py       — Face extraction using RetinaFace
│   └── multi_modal_model.py — Final multi-modal model architecture
├── train.py               — Training script
├── eval.py                — Evaluation script
├── config.py              — Model and training configurations
├── env.sh                 — Shell script for virtual environment setup
└── README.md              — Project documentation
```

## Setup

1. **Environment Preparation**  

   Set up the required environment by running:
   ```bash
   source env.sh
   ```

2. **Data Preprocessing**

   Convert emotion labels to numeric form:
   ```bash
   python csv_convert.py
   ```
   
   Remove entries with inaccessible videos:
   ```bash
   python clean_csv.py
   ```
3. **Training**

   Start model training
   ```bash
   python train.py
   ```
4. **Evaluation**

   Evaluate the trained model:
   ```bash
   python eval.py
   ```

## Key Components

- **Face Feature Extraction**  
  `modules/mobilefacenet.py` — MobileFaceNet-based CNN model for extracting facial features from video frames.

- **Audio Feature Extraction**  
  `modules/audio_model.py` — CNN model for extracting audio features from spectrogram representations.

- **Face Detection**  
  `modules/face_detect.py` — RetinaFace-based face detector to crop face regions from frames.

- **Multi-Modal Fusion**  
  `modules/multi_modal_model.py` — Fusion model combining face and audio features for final emotion classification.

- **Data Handling**  
  `data/meld_dataloader.py` — Data loader for the MELD dataset.  
  `data/collate_fn.py` — Custom collate function to handle batching of variable-length sequences.

- **Configuration**  
  `config.py` — Defines model hyperparameters, data paths, and training settings.

## Notes

- The project uses MobileFaceNet for lightweight facial feature extraction and CNN-based audio feature processing.
- Facial regions are automatically detected and cropped using RetinaFace before feeding into the model.
- Data augmentation techniques and normalization are integrated into the data loading pipeline.
- Designed for real-time, low-latency emotion recognition applications.
- Ensure the MELD dataset is correctly prepared in the specified directory structure before starting training.



   
   
