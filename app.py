# app.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

# ✅ Load fine-tuned model safely
MODEL_PATH = "bert_sentiment_model"  # folder containing config.json + pytorch_model.bin

# Use torch_dtype and device_map to prevent meta-tensor error
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto"  # ✅ let HF decide how to load safely
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# ✅ Label mapping for 3-class model
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# ----------------------------------------------------------
# 🧠 Streamlit UI
# ----------------------------------------------------------
st.title("🧠 BERT Sentiment Classifier")
st.write("Fine-tuned Encoder-only Transformer for Customer Feedback Classification")

user_text = st.text_area("Enter feedback:", height=150)

if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # ✅ move inputs to same device as model
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        sentiment = id2label[pred]

        if sentiment == "positive":
            st.success(f"Predicted Sentiment: **{sentiment.upper()}** 😄")
        elif sentiment == "neutral":
            st.info(f"Predicted Sentiment: **{sentiment.upper()}** 😐")
        else:
            st.error(f"Predicted Sentiment: **{sentiment.upper()}** 😞")


# Footer
st.markdown("---")
st.caption("Developed using **BERT-base-uncased** fine-tuned on Customer Feedback Dataset.")
