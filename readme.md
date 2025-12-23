# 📄 Resume / CV AI Analyzer

An AI-powered web application for analyzing resumes/CVs using a **practical machine learning approach**, combining classical NLP models with modern LLM-based reasoning to provide meaningful, explainable feedback.

---

## 🚀 Project Overview

Resume/CV AI Analyzer is built to help job seekers evaluate their resumes objectively before submitting them to recruiters.  
The system performs automated screening using **machine learning models** and enriches the results with **LLM-powered insights**.

A key takeaway from this project is learning that **simpler models can outperform complex ones** when datasets are limited or structured — avoiding unnecessary overengineering.

---

## 🎯 Key Features

- 📂 Upload resume files (`PDF` / `DOCX`)
- 🤖 Resume screening using trained ML models
- 📊 Role classification & acceptance prediction
- 🧠 AI-generated feedback and improvement suggestions
- 💬 Chatbot that answers questions based on resume analysis
- 📈 Model evaluation and performance transparency

---

## 🧠 Model & Approach

### 🔬 Initial Exploration
The project initially experimented with **DistilBERT** for resume classification.  
While powerful, it proved to be **overkill** for the available dataset:
- High computational cost
- Difficult tuning
- No significant performance gain

### ✅ Final ML Pipeline
The final solution uses **lighter, more interpretable models**:

- **Logistic Regression**
- **Multinomial Naive Bayes**
- **TF-IDF Vectorization**
- **SMOTE** for handling imbalanced datasets

This approach resulted in:
- Better stability
- Faster inference
- Easier interpretability
- More consistent evaluation metrics

---

## 🏗️ System Architecture

```bash
Resume Upload
↓
Text Extraction (PDF / DOCX)
↓
TF-IDF Vectorization
↓
ML Models
├─ Role Classification
└─ Acceptance Prediction
↓
Sentence Transformer (Similarity)
↓
LLM Review & Recommendation
↓
Streamlit UI Output
```

---

## 🧰 Tech Stack

### Machine Learning
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- SMOTE
- Sentence Transformers (`all-MiniLM-L6-v2`)

### LLM & AI
- Google Gemini API
- LangChain

### Backend & App
- Python
- Streamlit
- Joblib / Pickle

### Frontend
- Streamlit UI

### Datasets
- Role Classification: [kaggle.com | snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- Acceptance Prediction: [huggingface.co | AzharAli05/Resume-Screening-Dataset](https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset)

---

## 📊 Model Evaluation

The application provides:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

This transparency allows users to understand **why** a resume is rated a certain way.

---

## 📦 Installation & Local Setup

```bash
git clone https://github.com/RaffiAkhdilputra/resume-cv-analyzer.git
cd resume-cv-analyzer
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 👤 Author

### 👨 Raffi Akhdilputra
Informatics Engineering Student

### 👩 Aisyah Naurotul Athifah
System Information Student

### 👨 Ilham Satria Difta
Statistics Student

### 📫 Feel free to explore, fork, or reach out for discussion!

---

## ⭐ Acknowledgment

This project was built as part of hands-on learning in machine learning, NLP, and applied AI engineering. Focusing on real-world trade-offs, not just model complexity.

---

🔗 **Live Demo**:  
https://resumecv-ai-analyzer.streamlit.app/
