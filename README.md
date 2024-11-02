# MentalHealthGPT

## Overview

**MentalHealthGPT** is an AI-driven application designed to support mental health counselors by analyzing the tone, or “vibe,” of client conversations. This project integrates advanced NLP models to classify emotional tones, with the goal of providing counselors with insights into their client’s emotional state. It uses:

<p align="center">
  <img src="https://github.com/kianpaya/MentalHealthGPT/blob/9ff53265a986d21ac1eef3b627442a551a31eb6b/Images/Legacy%20App.jpg" alt="MentalHealthGPT Interface">
</p>

<p align="center"><i>Interactive interface of the MentalHealthGPT app, designed to analyze conversation tone.</i></p>

- **BERT Model** for machine learning classification, detecting the tone of text input to identify emotional categories.
- **GPT-based Fine-Tuning** using OpenAI’s API to enhance BERT model predictions, resulting in more accurate and contextual responses.

The application’s user-friendly interface is built with Streamlit and is hosted on **Hugging Face Spaces** for easy access.

---

## Project Components

### 1. Machine Learning Models
- **BERT-based Classification**: The model detects the emotional tone of the conversation. By classifying sentiments such as supportiveness, empathy, or frustration, the app provides valuable feedback to counselors.
- **Fine-Tuning with GPT**: An additional layer of customization is applied using the OpenAI API, which improves the BERT model’s contextual understanding by allowing the `egpt` module to adjust responses based on specific mental health needs.

### 2. Data Flow and Integration
- The project uses Hugging Face’s transformer library for the BERT model and OpenAI API for fine-tuning, providing a robust framework for processing and enhancing emotional tone detection in mental health conversations.

### 3. User Interface
- A **Streamlit-based UI** makes it easy for users to interact with the application, input chat text, and receive immediate feedback on the emotional tone. Hosted on **Hugging Face Spaces**, the UI is accessible via a simple web link.

---

## Installation

To run this project locally, follow these steps:

### Prerequisites
Ensure that Python is installed on your machine. You’ll also need API keys for **OpenAI** and **Hugging Face**.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kianpaya/MentalHealthGPT.git
   cd MentalHealthGPT
Set Up a Virtual Environment:

bash
Copy code
python3 -m venv env
source env/bin/activate  # For macOS/Linux
env\Scripts\activate     # For Windows
Install Required Packages: Use the requirements.txt file to install dependencies:

bash
Copy code
pip install -r requirements.txt
This will install key libraries such as torch, transformers, streamlit, openai, and sentence-transformers.

Log in to Hugging Face CLI:

Create a Hugging Face account if you don’t have one and generate an access token.
Log in via CLI:
bash
Copy code
huggingface-cli login
Set Up OpenAI API Key:

Obtain an API key from OpenAI and set it as an environment variable:
bash
Copy code
export OPENAI_API_KEY='your_openai_api_key'
Running the Application
To start the app locally:

bash
Copy code
streamlit run app.py
Usage
Enter Chat Text: Users input a conversation snippet to analyze.
Receive Emotional Tone: The app uses BERT to classify the tone (e.g., empathy, supportiveness).
Enhanced Response: The fine-tuned GPT model refines the response for more specific counseling insights.
Deployment on Hugging Face Spaces
The project is deployed on Hugging Face Spaces, making it accessible through a public link without requiring local setup. This setup leverages Hugging Face’s infrastructure for seamless, on-demand usage.

To access the application online:

Visit the Hugging Face Space: MentalHealthGPT on Hugging Face.
Use the Streamlit UI to interact with the app.
Requirements
To ensure smooth operation, all necessary libraries are listed in requirements.txt:

plaintext
Copy code
torch
torchvision
transformers
scikit-learn
numpy
alive-progress
openai==0.28
pandas
sentence-transformers
streamlit
Project Structure
plaintext
Copy code
MentalHealthGPT/
├── app.py                  # Main Streamlit application file
├── egpt.py                 # Fine-tuning module using OpenAI API
├── etal.py                 # BERT classification module for vibe detection
├── requirements.txt        # List of dependencies
├── Images/
│   └── Legacy App.jpg      # Sample app image
├── README.md               # Project README documentation
└── ...
Future Directions
Potential future updates for MentalHealthGPT include:

Enhanced Model Training: Further fine-tuning the BERT model for more nuanced emotion classification.
Expanded Dataset: Integrating additional mental health datasets to improve model accuracy.
Additional Deployment Options: Considering cloud-based hosting options to handle higher usage.
Acknowledgments
This project leverages models and tools from:

Hugging Face: For providing BERT and other NLP models.
OpenAI: For enabling advanced fine-tuning through the GPT API.
Streamlit: For creating a user-friendly interface to interact with the models.
