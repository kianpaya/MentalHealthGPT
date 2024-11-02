MentalHealthGPT
Overview
MentalHealthGPT is an AI tool for mental health counselors that analyzes conversation tone and generates context-sensitive responses. It uses BERT for tone classification and GPT (via OpenAI API) for fine-tuning and response generation, with a user-friendly Streamlit interface hosted on Hugging Face Spaces.

<p align="center"> <img src="https://github.com/kianpaya/MentalHealthGPT/blob/9ff53265a986d21ac1eef3b627442a551a31eb6b/Images/Legacy%20App.jpg" width="600" alt="MentalHealthGPT Interface"> </p> <p align="center"><i>MentalHealthGPT interface: classifies conversation tone and provides counseling guidance.</i></p>
Project Components
BERT Classification: Analyzes text tone to categorize emotional states like empathy, support, or frustration.
GPT Fine-Tuning: Refines BERT’s outputs using OpenAI’s API to generate nuanced, context-appropriate responses.
Streamlit UI: Provides an interactive platform, hosted on Hugging Face Spaces, where users input text to receive real-time tone analysis and response.
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/kianpaya/MentalHealthGPT.git
cd MentalHealthGPT
Set Up a Virtual Environment:

bash
Copy code
python3 -m venv env
source env/bin/activate      # For macOS/Linux
env\Scripts\activate         # For Windows
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Authenticate with Hugging Face and OpenAI:

Log in to Hugging Face:
bash
Copy code
huggingface-cli login
Set OpenAI API Key:
bash
Copy code
export OPENAI_API_KEY='your_openai_api_key'
Run the Application:

bash
Copy code
streamlit run app.py
Usage
Input Text: Enter a snippet of conversation.
Analyze Tone: BERT classifies the tone (e.g., empathetic, supportive).
Generate Response: GPT refines the response based on tone.
Deployment on Hugging Face Spaces
This app is deployed on Hugging Face Spaces for easy access. For deployment:

Prepare Repository:

Ensure app.py and requirements.txt are complete.
Push the project to GitHub.
Create a Space on Hugging Face:

Select Streamlit as the framework.
Link to your GitHub repository.
Set API Keys in Space Settings:

Add OPENAI_API_KEY as a secret for secure access.
Hugging Face will automatically deploy and generate a public link, updating with each GitHub push.

Project Structure
plaintext
Copy code
MentalHealthGPT/
├── app.py                  # Main application file
├── egpt.py                 # GPT fine-tuning module
├── etal.py                 # BERT classification module
├── requirements.txt        # Project dependencies
├── Images/
│   └── Legacy App.jpg      # App image
└── README.md               # Project README
Requirements
torch
torchvision
transformers
scikit-learn
numpy
openai
pandas
sentence-transformers
streamlit
Future Enhancements
Expanded Datasets: Integrate more mental health datasets for accuracy.
Model Refinement: Further fine-tune BERT for nuanced tone detection.
Deployment Options: Explore additional cloud hosting for higher traffic.
