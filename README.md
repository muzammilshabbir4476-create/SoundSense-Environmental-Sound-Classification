
# SoundSense: Environmental Sound Classification Using Deep Learning

## Project Overview
SoundSense is a deep learning project that classifies environmental audio into 10 sound categories using pretrained YAMNet audio embeddings and a custom neural network classifier.

## Objectives
- Classify environmental sounds from audio recordings.
- Use transfer learning with pretrained YAMNet embeddings.
- Evaluate the model on a held-out test set.
- Provide a foundation for an audio prediction web application.

## Dataset
This project uses the ESC-50 environmental sound dataset.

Official dataset: https://github.com/karolpiczak/ESC-50

Selected classes:
1. Dog
2. Cat
3. Chirping birds
4. Rain
5. Thunderstorm
6. Car horn
7. Siren
8. Airplane
9. Sea waves
10. Crying baby

The selected subset contains 400 audio clips, with 40 clips per category.

## Model Architecture
1. Audio is loaded as mono audio at 16 kHz.
2. Pretrained YAMNet extracts audio embeddings of 1024 features.
3. Mean pooling summarizes frame-level embeddings.
4. A custom dense neural network predicts one of the 10 classes.

## Dataset Split
- Training: 240 clips
- Validation: 80 clips
- Testing: 80 clips

The split uses ESC-50 folds to keep the test set separate from training.

## Results
Test accuracy: 90% (72 correct predictions out of 80 test clips).

This result is based on a relatively small test set and may not represent performance on all real-world recordings.

## Model Files
The `model/` directory contains:
- `soundsense_yamnet_classifier.keras`
- `soundsense_normalization.npz`
- `soundsense_categories.json`

YAMNet is also required for inference and is loaded from TensorFlow Hub.

## How to Run
The training and inference notebooks will be provided in the `notebooks/` directory.

Required packages include TensorFlow, TensorFlow Hub, librosa, NumPy, and scikit-learn.

## Future Work
- Deploy an interactive web application.
- Evaluate performance on more real-world audio samples.
- Improve model robustness and user experience.

## Acknowledgements
ESC-50 dataset: https://github.com/karolpiczak/ESC-50
YAMNet: https://tfhub.dev/google/yamnet/1
