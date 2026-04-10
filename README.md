# Sentiment-Aware AI Chatbot 🤖

A local, privacy-focused Generative AI chatbot that detects user sentiment and adapts its conversational tone in real-time. This project uses TinyLlama for text generation and DistilBERT for sentiment analysis, running entirely on your local machine.

## 🌟 Key Features
- **Sentiment Awareness**: Detects Positive, Negative, or Neutral emotions to guide AI responses.
- **Local-First**: Runs offline using TinyLlama-1.1B; no API keys or cloud costs required.
- **Privacy-Centric**: All conversations and model inferences stay on your device.
- **Modern UI**: Built with Streamlit for a clean, responsive chat experience.

## 🛠️ Tech Stack
- **Language**: Python 3.13+
- **LLM**: TinyLlama-1.1B-Chat-v1.0
- **Sentiment Model**: DistilBERT (SST-2)
- **Frameworks**: Transformers, PyTorch, Streamlit

## 🚀 Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/Sentiment-Aware-Chatbot.git
cd Sentiment-Aware-Chatbot
python -m venv venv
# Activate your venv (Windows)
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the Model
Run the utility script to fetch the TinyLlama weights from Hugging Face:
```bash
python download_model.py
```

### 3. Run the Chatbot
Launch the Streamlit interface:
```bash
streamlit run app.py
```

## 📂 Project Structure
```text
sentiment-chatbot/
├── app.py              # Main Streamlit application
├── download_model.py   # Utility script to fetch LLM weights
├── requirements.txt    # Project dependencies
├── models/             # Local storage for model weights (ignored by git)
└── venv/               # Virtual environment (ignored by git)
```

## 📝 License
MIT License
