# 📰 Text Mining & Analysis on News Podcasts (NBC & NYT)

A complete text-mining and NLP project analyzing **NBC News** and **New York Times** podcast transcripts using custom **inverted & positional indexes**, **TF-IDF**, **word co-occurrence networks**, and **sentiment analysis (BERT + spaCy)**.


---

## 🎯 Project Overview

This project extracts insights from news podcast transcripts by applying text mining and natural language processing techniques.  
We focused on identifying:

- Major **news topics**
- Differences in coverage between **NBC vs NYT**
- **Sentiment variation** across episodes
- Frequently co-occurring terms and clusters
- Vocabulary style and topic emphasis
- Efficient retrieval using custom **information retrieval indexes**

The goal is to understand how different news outlets frame stories and what linguistic patterns emerge across episodes.

---

## 📡 Data Description

### Sources:
- NBC News podcast transcripts  
- New York Times (NYT) podcast transcripts  

### Characteristics:
- Conversational, multi-speaker text  
- Length: 10–20 minutes per transcript  
- Contains filler words, repetitions, informal speech  

### Collection Process:
- Podcast episodes downloaded  
- Transcripts extracted + cleaned  
- Organized into separate NBC and NYT corpora  

---

## 🧹 Data Preprocessing

Performed using **spaCy** and **NLTK**:

- Tokenization  
- Lowercasing  
- Stopword removal  
- Lemmatization  
- Removal of filler words (“uh”, “yeah”, “you know”)  
- Cleaning special characters + non-English tokens  
- Sentence segmentation  

This created high-quality text suitable for further NLP tasks.

---

## 🗂️ Data Representation

Constructed:

- **Inverted Index** (term → document mapping)  
- **Positional Index** (term → positions within each transcript)  
- **Term Frequency tables**  
- **TF-IDF vectors**  
- **Word co-occurrence matrix**  
- **Sentence embeddings** for BERT sentiment analysis  

---

## 🧠 Models & Techniques Used

### 1️⃣ Inverted Index
Custom implementation for keyword-based search and retrieval.

**Outputs include:**
- Most common terms in NBC & NYT  
- Query-based term lookup  
- Evidence of differing vocabulary styles  

---

### 2️⃣ Positional Index
Stores the order and location of terms in transcripts.

**Outputs include:**
- Phrase search (“border security”, “climate change”)  
- Term proximity analysis  
- Comparison of language structure between NBC vs NYT  

---

### 3️⃣ Term Frequency & TF-IDF Analysis
Computed raw term frequencies and TF-IDF weights.

**Findings:**
- NBC emphasizes breaking news & domestic issues  
- NYT emphasizes global affairs, deeper analysis, policy topics  

---

### 4️⃣ Word Co-Occurrence Network
Built using **NetworkX**.

Reveals:

- Central concepts  
- Strongly connected word clusters  
- Key themes within each news source  

**Patterns:**
- NBC → short-form, high-frequency domestic clusters  
- NYT → analytical, globally oriented clusters  

---

### 5️⃣ Sentiment Analysis — BERT
Used pretrained BERT model to determine:

- Positive/negative/neutral scores  
- Episode-level sentiment patterns  
- Differences in emotional tone between outlets  

---

### 6️⃣ Sentiment Analysis — spaCy
Rule-based model providing:

- Polarity scoring  
- Neutral vs opinionated phrasing  
- Validation against BERT’s predictions  

---

## 📊 Visualization Outputs

Generated visualizations for:

- Term frequency bar charts  
- TF-IDF key terms  
- Word co-occurrence networks  
- Sentiment distribution graphs  
- Topic clouds  
- Inverted/positional index retrieval examples  

All results were compared for **NBC vs NYT**.

---

## 🏛️ System Architecture

Data Collection<br>
↓<br>
Preprocessing<br>
↓<br>
Index Construction (Inverted + Positional)<br>
↓<br>
Feature Extraction (TF, TF-IDF)<br>
↓<br>
Word Co-Occurrence Analysis<br>
↓<br>
Sentiment Analysis (BERT + spaCy)<br>
↓<br>
Visualization<br>
↓<br>
Insights & Comparison


---

## 🔍 Key Findings

- **NYT** uses a more diverse and analytical vocabulary.  
- **NBC** focuses on fast-paced, domestic breaking news.  
- **Sentiment trends:**
  - NYT → more neutral and explanatory  
  - NBC → more variation depending on topic  
- **Co-occurrence networks:**
  - NYT → global, political, economic clusters  
  - NBC → social issues, policy, rapid updates  
- **Indexing reveals**:
  - Key phrases recur differently across sources  
  - NBC uses tighter keyword clusters  
  - NYT uses longer, more descriptive phrasing  

---

## 🧰 Tools & Technologies

- Python  
- spaCy  
- NLTK  
- scikit-learn (TF-IDF)  
- Transformers (BERT sentiment model)    
- Pandas / NumPy  
- Matplotlib / Seaborn  

---

## 📂 Repository Structure




---

## 🖥️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/News_Podcast_Text_Mining
