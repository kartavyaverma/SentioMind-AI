import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/tinyllama"

st.set_page_config(page_title="Sentiment Chatbot", layout="wide")

st.title("SentioMind AI")

#Loading Model
@st.cache_resource
def load_models():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    return sentiment_model, tokenizer, model


sentiment_model, tokenizer, model = load_models()


def get_sentiment(text):
    result = sentiment_model(text)[0]
    label = result['label']
    score = result['score']

    if score < 0.6:
        return "NEUTRAL"
    return label

def build_prompt(user_input, sentiment):

    if sentiment == "POSITIVE":
        system = "You are a friendly and enthusiastic assistant."
    elif sentiment == "NEGATIVE":
        system = "You are a calm and empathetic assistant."
    else:
        system = "You are a helpful assistant."

    prompt = f"""
<|system|>
{system}
<|user|>
{user_input}
<|assistant|>
"""
    return prompt

def generate_response(user_input):

    sentiment = get_sentiment(user_input)
    prompt = build_prompt(user_input, sentiment)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()

    return sentiment, response


# UI
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("About")
    st.write("Local TinyLlama chatbot with sentiment awareness.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

# Input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    with st.spinner("Thinking..."):
        sentiment, response = generate_response(user_input)

    final_response = f"{response}\n\n🧠 *Sentiment: {sentiment}*"

    st.session_state.messages.append(("assistant", final_response))
    st.rerun()