# Multimodal Prediction of Memecoin Market Success on Pump.fun

> 📌 This repository contains the code and materials for my undergraduate **Signature Work** project.

## Overview

This project investigates whether **early-stage multimodal information** can be used to predict which memecoins created on **Pump.fun** will achieve relatively higher market success after launch.

Using the open **Coin-Meme** dataset by Long, Wong, and Cai (2025), the project builds an interpretable machine learning pipeline that combines:

- **Textual descriptions** of each token  
- **Logo images** (placeholder for future work in this repo)  
- **User comments** and early community sentiment  
- **Basic financial and timing variables** (e.g. market entry time, peak market capitalisation)

The goal is not to provide trading advice, but to explore how culture, community, and early market signals interact in the highly speculative memecoin ecosystem.

## Research Questions

This Signature Work is guided by the following research questions:

- **RQ1:** Can information that is observable at or shortly after launch be used to predict which memecoins created on Pump.fun will achieve relatively higher market success?

- **RQ2:** Does combining multiple types of information (text, community signals, and financial data) improve predictive performance compared with models that rely on a single source of information?

- **RQ3:** Which broad aspects of a memecoin—such as its narrative, early community reaction, and initial market conditions—appear most closely associated with subsequent market success?

## Methods

### Data

- **Base dataset:** Coin-Meme dataset (Long, Wong, & Cai, 2025), containing:
  - Token-level metadata and descriptions  
  - Pump.fun logo images  
  - User comments  
  - Financial variables such as market entry time and market capitalisation  

- **Local files used in this repo:**
  - `data/tokens.csv` — token-level data (IDs, descriptions, max market cap, market entry time, etc.)  
  - `data/comments.csv` — user-level comments mapped to token IDs  

> ⚠️ The data files in this repo are used for academic purposes only. Please refer to the original Coin-Meme paper for dataset details and citation.

### Target Variable

- Define **market success** as a binary label:
  - `HighCap` (1): tokens with above-median peak market capitalisation  
  - `LowCap` (0): tokens with below-median peak market capitalisation  

Only information observable **at or shortly after launch** is used as input to the models.

### Features

- **Textual features (descriptions)**
  - Cleaned and vectorised using **TF–IDF** (unigrams and bigrams, capped vocabulary size).
- **Community features (comments)**
  - Aggregate statistics per token:
    - number of comments, number of unique users  
    - mean VADER sentiment score  
    - sentiment standard deviation  
    - positive / negative sentiment ratios  
- **Financial & timing features**
  - Market entry time (Pump.fun → Raydium)  
  - Simple timing features (e.g. hour-of-day, etc., if available)  
- **Image features**
  - In this version of the repo, image features are implemented as a placeholder (zero vectors) to keep the pipeline extensible for future work.

### Models

Implemented in `scikit-learn`:

- Logistic Regression (linear baseline)  
- Random Forest  
- Gradient Boosting Classifier  

Data is split into **train / validation / test** with stratification:

- Train: 70%  
- Validation: 15%  
- Test: 15%  

### Evaluation

- Accuracy  
- F1-score (for the HighCap class)  
- ROC–AUC  

The best model on the validation set is then evaluated on the held-out test set.

---

## Repository Structure

```text
memecoin-success-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── tokens.csv         # token-level data
│   ├── comments.csv       # comment-level data (NOT uploaded yet due to size)
│   └── images/            # (optional) logo images for future image features (NOT uploaded yet due to size)
├── models/
│   └── best_model.joblib  # saved best-performing model (created after training)
└── src/
    └── main.py            # main training & evaluation script
