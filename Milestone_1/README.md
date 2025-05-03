# Arabic Transcript Analysis Pipeline 🚀

This repository contains the code and results from **Milestone 1** of the CSEN1076: Natural Language Processing and Information Retrieval course. Our goal was to build a preprocessing and analysis pipeline for Arabic YouTube transcripts (ElDa7ee7 channel) and explore various NLP tasks and visualizations.

## 📁 Project Structure

* `data/` – Raw and cleaned transcript files 🗂️
* `notebooks/` – Jupyter notebooks for preprocessing and analysis 📓
* `models/` – Saved Hugging Face model checkpoints 💾

## 🔄 Preprocessing Pipeline

1. **🧹 Tidying Up**

   * Removed timestamps and non-text noise (hashtags, emojis, etc.) 🗑️
   * Attempted spelling correction with Ghalatawi (did not yield expected improvements) 📝
2. **🔠 Tokenization & Sentence Splitting**

   * Applied SBERT embeddings + cosine similarity threshold to segment transcripts into coherent sentences ✂️
3. **🌐 Language Translation**

   * Tried translating Egyptian Arabic to English for comparative analysis (translation inconsistencies limited its utility) 🔄
4. **✂️ Stopword Removal & Dediacritization**

   * Combined NLTK, TASHAPHYNE, and custom stopword lists 🛠️
   * Removed diacritics to normalize text 🔤

## 🧠 NLP Analysis Tasks

* **😊😠 Sentiment Analysis**
  Used `bert-base-arabic-camel-msa-sentiment` from CAMeL Lab. Model over-classified negative labels due to dialect-specific word usage.

* **📚 Topic Classification**
  Fine-tuned `bert-base-arabertv2` from Aubmind Lab on multiple categories (culture, politics, tech, etc.). Frequent misclassification as "religion" due to common religious terms.

* **🤨 Sarcasm Detection**
  Fine-tuned on Twitter dataset; performance was limited by dataset size and dialectal variation.

* **📰 Text Summarization**
  Evaluated `mbert2mbert-arabic-text-summarization`; produced hallucinations on long transcripts.

## 📊 Visualization & Insights

* **📑 Basic Statistics**
  Computed word counts, unique tokens, sentence counts, and TF–IDF scores.

* **☁️ Word Cloud**
  Generated before/after stopword removal using `word_cloud`, `arabic-reshaper`, and `python-bidi`.

* **🔢 Most Frequent N-grams**
  Identified top 3-grams; revealed highly repetitive filler phrases acting as outliers.

* **🚫 Outlier Removal & 🏷️ NER**
  Removed common fillers; applied CAMeL Labs NER to extract top nouns (e.g., historical entities, country names).

* **🔗 Content Similarity**
  Built cosine similarity matrix across videos; found limited overlap with occasional thematic repeats.

* **❤️👁️ Engagement Analysis**

  * **Likes vs. Category** – Political videos showed high average likes; small sample sizes noted.
  * **Likes vs. Description Length** – Positive correlation between word count and likes.
  * **Views vs. Category** – Politics and Culture attracted higher views.
  * **Views vs. Transcript Length** – Longer transcripts correlated with more views.
  * **Views vs. Sarcasm Frequency** – Spikes in views for episodes with very high sarcasm levels.

## 🚀 Potential Follow-Up Tasks

* Regression modeling to predict views/likes from transcript features 📈
* Improved dialectal sentiment and sarcasm models with larger labeled data 🗣️
* Topic modeling (LDA) on cleaned transcripts 🧐
* Fine-tuning sequence-to-sequence summarization on domain-specific corpus ✍️

## 🎯 Conclusion

This milestone demonstrated the challenges of Arabic NLP (dialectal variance, model biases) and provided foundational insights into ElDa7ee7’s content and audience engagement. The pipeline is modular and can be extended for downstream predictive and generative tasks.

---

*Prepared by Mazen Soliman & Mohamed Shamekh* 😊
