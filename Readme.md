🧠 Encoder-Only (BERT) — Customer Feedback Classification

This repository contains our implementation of Task 1 from Project 03: Fine-Tuning Transformer Architectures for Real-Time NLP Applications.
The task focuses on fine-tuning a pre-trained BERT model for sentiment classification on customer feedback data.
Each feedback entry is classified as Positive, Neutral, or Negative.

🎯 Objective

The goal was to adapt an Encoder-Only Transformer (BERT-base-uncased) to analyze customer feedback and predict sentiment polarity with high accuracy.

📂 Dataset

Source: Kaggle – Customer Feedback Dataset

Classes: Positive | Neutral | Negative

Preprocessing: Cleaning, label encoding, tokenization with Hugging Face Tokenizer

⚙️ Model Details
Component	Description
Architecture	BERT-base-uncased (Encoder-Only Transformer)
Task	Sequence Classification
Framework	PyTorch + Hugging Face Transformers
Training Epochs	8
Batch Size	16 (train) / 32 (eval)
Optimizer	AdamW (learning rate = 2e-5)
Evaluation Metrics	Accuracy and F1-Macro
📊 Performance Summary
Metric	Value
Accuracy	1.00
F1-Macro	1.00

Confusion Matrix

[[9, 0, 0],
 [0, 1, 0],
 [0, 0, 11]]

🧩 Example Predictions
Customer Feedback	Predicted Sentiment
“The product quality is excellent and delivery was fast.”	Positive
“It’s okay, not great but not terrible either.”	Neutral
“Worst service ever, I’m extremely disappointed.”	Negative
💻 Usage

To run the Streamlit application locally:

pip install -r requirements.txt
streamlit run app.py

🧠 Project Deliverables

✅ Preprocessing and tokenization scripts

✅ Training and validation pipeline

✅ Evaluation (metrics + confusion matrix)

✅ Example predictions in real-time via Streamlit interface

👥 Contributors

Muhammad Usman Awan
Omima

🔗 Repository Link

[https://github.com/ayan364/Encoder-Only-BERT-Sentiment-Classification](https://github.com/Usman-Ifty/Encoder-Only-BERT-Sentiment-Classification?tab=readme-ov-file)


