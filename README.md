# Fake-News-Prediction-Model
An end-to-end NLP and machine learning pipeline built in Google Colab to classify real vs. fake news articles using Logistic Regression and Decision Trees. Adapted from a GeeksforGeeks tutorial with custom debugging optimization for data shuffling, text preprocessing, and matrix evaluations.
# 📰 Fake News Prediction Model

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Made%20in-Google%20Colab-F9AB00.svg)](https://colab.research.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

A Python-based NLP and machine learning pipeline that classifies news articles as **Real** or **Fake**. Built end-to-end inside a Google Colab notebook, from raw text to evaluated model.

---

## 📖 Table of Contents
- [Overview](#-project-overview)
- [Core Features](#core-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Data Processing & Engineering](#-data-processing--engineering)
- [Debugging & Adaptation](#-real-world-debugging--adaptation)
- [Evaluation & Diagnostics](#-evaluation--diagnostics)
- [Acknowledgments](#-acknowledgments--attributions)
- [License](#-license)

---

## 📌 Project Overview
The goal of this project is a robust binary text classification engine. Using text normalization and classical ML algorithms, the pipeline learns the language patterns inherent to deceptive news reporting to systematically flag misinformation.

### Core Features
- Full data ingestion, automated row shuffling, and clean index restructuring
- Textual feature engineering via **TF-IDF** (`TfidfVectorizer`)
- Comparative modeling across multiple estimators (**Logistic Regression** and **Decision Tree Classifier**)
- Visualization suite for vocabulary distribution (WordClouds, bar charts) and model evaluation (Seaborn confusion matrices)

---

## 🛠 Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3 |
| Environment | Google Colab |
| ML / Modeling | scikit-learn |
| Text Processing | NLTK, regex |
| Data Handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |

---

## 🚀 Getting Started

1. Open the notebook in Google Colab (click the badge above or upload the `.ipynb` directly).
2. Run the cells in order — dependencies install automatically at the top of the notebook.
3. Provide your dataset (or use the included sample) and run the pipeline end-to-end, from preprocessing through evaluation.

> This project was built and tested in Colab, so a local environment isn't required — just a Google account.

---

## 🔬 Data Processing & Engineering
1. **Shuffling & Reset** — Dataset rows are fully randomized with `.sample(frac=1)` to mix labels, paired with clean index restructuring to protect pipeline memory tracking.
2. **Text Optimization** — Custom regular expressions strip structural formatting and punctuation. Tokens are lowercased and filtered against standard `NLTK` English stop words.
3. **Matrix Generation** — Words are converted into numerical vectors using TF-IDF weights, capturing contextual relevance rather than raw frequency counts.

---

## 🔧 Real-World Debugging & Adaptation
Several logical and structural bugs were resolved during development to ensure pipeline integrity:

- **Pandas DataFrame Indexing** — Fixed a structural crash where dropping the native index threw a `KeyError`, resolved with a safer chained approach: `.reset_index(drop=True)`.
- **String Join Logic & Space Collapses** — Fixed a preprocessing bug where tokens were compressed using `''.join()` with no whitespace boundary. Correcting this to `' '.join()` prevented vocabulary words from merging into broken text streams.
- **Canvas Allocation & Layout Failures** — Addressed Colab's `UserWarning: Tight layout not applied` by reconfiguring `matplotlib` with manual bounds (`subplots_adjust`), preventing long axis labels from collapsing the plot canvas.

---

## 📊 Evaluation & Diagnostics
- **Logistic Regression** — Stable learning behavior, producing a reliable and highly generalizable accuracy baseline on unseen data.
- **Decision Tree Classifier** — Achieved a perfect training fit but showed classic **overfitting** under unconstrained tree depth. Variance is visualized in an integrated Seaborn confusion matrix with percentage distributions.

---

## 📜 Acknowledgments & Attributions
- **Learning Source** — Implementation inspired by and adapted from structural concepts in a project tutorial by **GeeksforGeeks**... https://www.geeksforgeeks.org/machine-learning/fake-news-detection-using-machine-learning/
- **Modifications** — Base concepts were extended to fix operational pipeline bugs, update legacy syntax, and add custom diagnostic plots.

---

## ⚖️ License
This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
