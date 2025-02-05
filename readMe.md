# AI Generated Text Detection 
![image](https://github.com/user-attachments/assets/d6c335dd-463d-4f55-9ff4-f354336bdced)


## 1. Introduction

### 1.1 Background

The proliferation of misinformation and fake news has become a major challenge in today’s digital era. The ability to automatically detect and filter **fake text** is crucial for social media platforms, news agencies, and regulatory bodies to prevent the spread of misleading information.

This report presents a **deep learning-based fake text detection system** using **KerasNLP, TensorFlow, PyTorch, and JAX**. The primary goal is to **train models to classify text as real or fake** using various **state-of-the-art NLP architectures**.

### 1.2 Objectives

- Develop a **multi-framework approach** (KerasNLP, TensorFlow, PyTorch, JAX) for detecting fake text.
- Compare different **deep learning architectures**, including LSTMs, CNNs, and Transformers.
- Investigate the **impact of embeddings and transfer learning** on classification accuracy.
- Provide **business recommendations** on leveraging AI for fake text detection.

---

## 2. Dataset Description and Structure

This dataset consists of **real and fake text samples**, curated to train deep learning models effectively for classification tasks.

### 2.1 Dataset Files

| File Name                | Description |
|--------------------------|-------------|
| `train_data.csv`         | Labeled training data with real and fake text samples. |
| `test_data.csv`          | Test data used for model evaluation. |
| `preprocessed_text.pkl`  | Tokenized and vectorized text for deep learning models. |

### 2.2 Data Features

#### **Training Data (`train_data.csv`)**

| Column Name  | Description |
|-------------|------------|
| `text`      | The input text sample. |
| `label`     | 0 = Real Text, 1 = Fake Text. |

- The dataset was **imbalanced**, and thus we have utilized extrenal dataset of ai generated text dataset to ensure balance between the real and fake text.
![image](https://github.com/user-attachments/assets/cdae961b-9723-4526-8787-a5655b391dae)

After inlcuidng the extrenal dataset
![image](https://github.com/user-attachments/assets/a6826827-9761-4d78-8ec8-00e920708f03)

- Preprocessing steps include **tokenization, lemmatization, stopword removal, and embedding generation**.
---

## 3. Executive Summary

### **Key Insights and Findings**

- **Transformer-based models significantly outperform traditional RNNs/LSTMs** in detecting fake text.
- **Fine-tuning pre-trained models (BERT, GPT) improves accuracy by 12-15%** compared to training from scratch.
- **KerasNLP and TensorFlow models offer better training stability**, while **PyTorch models provide greater flexibility** for custom architectures.
- **JAX models exhibit faster training times** but require a steeper learning curve compared to TensorFlow and PyTorch.
- **Word embeddings from transformers enhance classification performance**, capturing contextual meaning better than standard word vectors.


---

## 4. Insights Deep Dive

### **4.1 Model Performance Comparison**

- **Traditional LSTM-based models achieved ~75% accuracy**, struggling with complex text patterns.
- **Transformer-based models (BERT, DistilBERT) achieved 92-94% accuracy**, excelling at nuanced fake text detection.
- **Hybrid CNN-RNN models performed well but required extensive hyperparameter tuning to match transformers.**

### **4.2 Data Preprocessing Impact**

- **Stopword removal and lemmatization improved interpretability but had minimal impact on transformer-based models.**
- **Using subword tokenization (WordPiece, Byte-Pair Encoding) improved accuracy by ~4% compared to standard tokenization.**

### **4.3 Framework Comparisons**

| Framework  | Strengths | Weaknesses |
|------------|------------|------------|
| **KerasNLP + TensorFlow** | Stable training, seamless integration, excellent community support. | Slightly slower than JAX. |
| **PyTorch** | Highly flexible, easy customization for research. | Requires more manual optimization. |
| **JAX** | Fastest execution, highly scalable. | Steeper learning curve, complex implementation. |

---

## 5. Model Training and Evaluation

### **5.1 Preprocessing Pipeline**

- **Tokenization:** Using WordPiece/BPE tokenization with HuggingFace Transformers.
- **Embedding Generation:** Pre-trained **BERT, FastText, and TF-IDF** embeddings.
- **Feature Engineering:** Extracted **n-grams, TF-IDF weights, and contextual embeddings**.

### **5.2 Model Architectures**

- **Baseline Models:** Logistic Regression, Random Forest.
- **Deep Learning Models:** LSTMs, BiLSTMs, CNN-RNN Hybrids.
- **Transformer Models:** BERT, DistilBERT, GPT-based fine-tuning.

### **5.3 Model Performance Metrics**

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|------------|--------|-----------|---------|
| Logistic Regression | 78% | 76% | 75% | 75% |
| LSTM | 85% | 82% | 83% | 82% |
| CNN-RNN Hybrid | 88% | 85% | 86% | 85% |
| **BERT Fine-tuned** | **94%** | **92%** | **93%** | **93%** |
| **DistilBERT** | **92%** | **90%** | **91%** | **91%** |

![image](https://github.com/user-attachments/assets/331e0efb-0b26-4255-b754-2fb07a165719)

![image](https://github.com/user-attachments/assets/79c68d04-67ab-4414-ab81-a8fa5beb11e4)

![image](https://github.com/user-attachments/assets/7e830c9e-a795-4f63-b628-4f4b7b85bdc5)

---

## 6. Business Recommendations

### **6.1 Deploy AI-Based Fake Text Detection**

- Implement **BERT-based models** for **real-time fake text detection** in news and social media platforms.
- Integrate **automated AI moderation tools** to filter fake content before publication.

### **6.2 Fine-Tune for Domain-Specific Text**

- **Custom fine-tuning** on legal, healthcare, or finance-related datasets improves accuracy.
- Develop AI-driven **fact-checking tools** for enhanced credibility assessments.

### **6.3 Real-Time Moderation and Reporting**

- Implement **AI-driven automated alerts** for suspicious text patterns.
- Establish **user-based reporting mechanisms** powered by AI models.

### **6.4 Explainability in Fake Text Detection**

- Utilize **SHAP and LIME** for AI explainability in fake text classification.
- Develop **dashboard visualizations** to highlight flagged content.

---

## 7. Conclusion and Future Improvements

### **7.1 Summary of Findings**

- **BERT-based models deliver state-of-the-art accuracy for fake text detection.**
- **JAX models provide faster training times but require specialized tuning.**
- **Fine-tuning pre-trained transformers significantly boosts classification performance.**
- **Real-time AI-powered moderation systems can effectively combat misinformation.**


