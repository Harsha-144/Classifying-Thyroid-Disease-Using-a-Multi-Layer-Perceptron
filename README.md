# Thyroid Disease Classification using Multi-Layer Perceptron (MLP)

This project predicts thyroid conditions using a custom-built neural network and provides health recommendations via **Google Gemini AI**.

## Overview

The application is trained to classify thyroid status into:
- **Normal**
- **Hypothyroid**
- **Hyperthyroid**

It uses:
- A **Multi-Layer Perceptron (MLP)** with one hidden layer
- Manual feature selection (21 features)
- Sigmoid activation
- Custom forward & backpropagation
- **Streamlit** for a user-friendly web interface
- **Gemini AI** for personalized health suggestions

## Project Structure

Thyroid-Disease-Classifier
├── app.py # Streamlit web app
├── train_model.py # Script to train the model and save weights
├── requirements.txt # All dependencies
├── data/
│ ├── ann-train.txt # Training dataset
│ └── ann-test.txt # Testing dataset
├── model/
│ └── utils.py # Sigmoid, weight loader, predictor


## How to Run

### Setup
1. **Clone the repo**  

   git clone https://github.com/your-username/thyroid-disease-classifier.git
   cd thyroid-disease-classifier
Install dependencies
  pip install -r requirements.txt
Train the model (once)
  python train_model.py
Run the Streamlit app
streamlit run app.py
Features
Predicts thyroid disease from 21 input values
Custom implementation of a feedforward MLP
Visual feedback with easy-to-use interface
Health suggestions via Google Gemini API
No cloud model dependency — works offline after training

Gemini API Integration
Gemini is used to generate condition-specific suggestions based on your model's output.
To use it:
Get your API key from Google AI Studio
It’s already integrated in app.py — just paste your key.
