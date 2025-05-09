# Author Prediction Neural Network

This repository contains a neural network model trained to predict the author of a text based on their writing style. The model is trained on entire novels from different authors and can classify new text samples.

## Features
- Text preprocessing and vectorization using TF-IDF
- Neural network architecture with dense layers and dropout for regularization
- Dataset creation from author text samples
- Training and evaluation on multi-author classification

The model extracts stylistic patterns from authors' works by analyzing chunks of text (approximately 40 words) and learns to distinguish between different writing styles.

## Implementation Details
- Uses TensorFlow for model creation and training
- Implements heavy dropout to prevent overfitting
- Processes text using TF-IDF vectorization
- Randomly samples text chunks from each author's works
