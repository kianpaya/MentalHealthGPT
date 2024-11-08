# MentalHealthGPT: AI-Powered Mental Health Support

**To those seeking understanding and connection in the digital realm.**

---

## Overview

**MentalHealthGPT** is an open-source, AI-driven application that provides mental health support through conversational AI. Built on GPT-4 with custom knowledge bases, this app aims to offer empathetic, supportive responses tailored to users' emotional needs. It integrates mental health datasets to generate relevant and helpful responses in real-time.

<p align="center">
  <img src="Images/Legacy_App.jpg" width="600" alt="MentalHealthGPT Interface">
</p>

<p align="center"><i>MentalHealthGPT app interface: Recognizes conversation tone and provides counseling guidance.</i></p>

---

## Project Goals

- Provide accessible mental health support via natural language understanding.
- Deliver contextually aware responses with AI tailored to user needs.
- Enhance mental health resources through an intuitive user interface.

---

## Features

- **Real-time Conversational AI**: Utilizes OpenAI’s GPT-4 API for interactive conversations.
- **Knowledge Base Integration**: Custom knowledge base created from mental health datasets for accurate responses.
- **Streamlit Interface**: User-friendly interface designed for easy access to mental health support.
- **Scalability and Deployment**: Hosted on Hugging Face Spaces for seamless access and testing.

---

## Demo & Hosting Link

- **Hugging Face Space**: [MentalHealthGPT on Hugging Face](https://huggingface.co/spaces/kianpaya/MentalHealthGPT)

> **Note**: Due to computational constraints, response times on Hugging Face may vary.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/kianpaya/MentalHealthGPT.git
cd MentalHealthGPT
```

### Set Up Environment and Install Dependencies

1. **Create a Virtual Environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```
2. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API Key**:
   - Set your OpenAI API key in Streamlit's secrets section if deploying or in `secrets.toml` locally.

    ```plaintext
    [secrets]
    openai_api_key = "your_openai_api_key_here"
    ```

---

## Usage

### Run the App Locally

To start the Streamlit app on your local machine:

```bash
streamlit run app.py
```

### Models

- **egpt Model**: Uses GPT-4 API for dynamic, context-aware conversational responses.
- **etal Model**: Provides general conversation responses without specialized knowledge base.

### Example Usage

1. **User Input**: "I feel really anxious about my upcoming exams."
2. **MentalHealthGPT Response**: "Preparing for exams can be stressful. Have you considered relaxation techniques or talking to someone for support?"

---

## Project Structure

- `app.py`: Main Streamlit application file.
- `egpt.py`: Code for the `egpt` model that integrates OpenAI’s GPT API and custom knowledge base.
- `etal.py`: A simpler model for generalized responses.
- `README.md`: Project documentation.
- `requirements.txt`: Lists Python dependencies.

---

## Model Details

### egpt Model
- **Embedding Model**: `all-MiniLM-L6-v2` for encoding and similarity matching in the knowledge base.
- **API**: Relies on OpenAI's GPT-4 API for generating responses, enhancing interaction with contextual accuracy.

### Dataset

- **Source**: Hugging Face [mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
- **Format**: JSON with `Context` and `Response` fields for question-answer pairs.
- **Integration**: Loaded and preprocessed for use in the custom knowledge base.

---

## Limitations

- **Model Training**: Currently trained on a single epoch, so accuracy may be limited.
- **Response Time**: API call times can vary due to Hugging Face computational limitations.

---

## Future Plans

Currently, the model uses GPT-4 Turbo without fine-tuning. In future iterations:
- **Fine-Tuning with OpenAI API**: Train on custom datasets to improve accuracy and relevance of responses.
- **Deployment**: Fine-tuned models hosted on OpenAI for enhanced performance.

---

## Performance

### Response Speed

- **Text Input**: Response times depend on Hugging Face’s available resources.
- **Scalability**: Designed for cloud hosting platforms for real-time interactions.
  
---

## Challenges

- **Accuracy in Tone Detection**: Tone detection is complex, and achieving accurate classification, especially in nuanced or ambiguous text, remains a challenge. Misclassification can lead to inappropriate or ineffective responses.
- **Dependency on External APIs**: Reliance on OpenAI’s API for GPT-based response generation can introduce latency and may be cost-prohibitive with large-scale usage.
- **Ethical Considerations**: As an AI-driven tool in mental health, there are ethical considerations around transparency, bias in model responses, and the potential impact of machine-generated responses on clients’ mental health.

---

## Future Plans

- **Enhanced Fine-Tuning**: Train a specialized mental health model with fine-tuned parameters on OpenAI.
- **Expanded Dataset**: Increase dataset diversity to improve response relevance.
- **Additional Models**: Implement more robust transformer architectures for specific mental health applications.

---

## Contributing

Contributions to MentalHealthGPT are welcome! Whether it’s improving response accuracy, optimizing the architecture, or adding new features, your efforts can help expand this project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Submit a pull request.

For any questions or suggestions, feel free to reach out.

---
 ## License

 - This project is licensed under the [MIT License](https://github.com/kianpaya/MentalHealthGPT/blob/main/LICENSE).
---

## Contact

- **Kian Paya** - [GitHub](https://github.com/kianpaya), [LinkedIn](https://www.linkedin.com/in/kianpaya/)
- **Feedback**: Feel free to try the app and provide feedback for improvements!

---

## Conclusion

**MentalHealthGPT** is a cutting-edge, AI-powered tool that brings empathy and understanding to digital mental health support. By integrating real-time tone analysis and context-aware responses, it offers mental health professionals a valuable resource to enhance client communication. As a versatile, accessible solution hosted on Hugging Face Spaces, MentalHealthGPT not only demonstrates the potential of AI in mental health but also invites a future where technology actively supports emotional well-being. Join us on this journey to make mental health support more effective, accessible, and compassionate, one conversation at a time!
