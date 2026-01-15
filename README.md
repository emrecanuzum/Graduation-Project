# The Identification of Satirical Fake News in Turkey 🇹🇷

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/NLP-Zemberek-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

This project focuses on the detection of misinformation and satirical fake news in the Turkish language using various Machine Learning and Deep Learning algorithms. 

The study distinguishes between genuine news from reliable sources and spurious/satirical content (e.g., Zaytung) using a dataset compiled specifically for this research.

## 📄 Abstract

The proliferation of misinformation is a significant issue in today's digital society. In the context of the Turkish language, distinguishing between serious journalism and satirical "fake" news requires robust computational approaches. This project implements and compares **XGBoost, Random Forest, Logistic Regression, LSTM, and CNN** models. 

Our findings indicate that **Logistic Regression outperformed other models with an accuracy of 91%**, followed closely by XGBoost and our custom CNN model.

## 👥 Authors

* **Şükrü Erim Sinal** - Istanbul Kultur University
* **Ahmet Kaan Memişoğlu** - Istanbul Kultur University
* **Emrecan Üzüm** - Istanbul Kultur University

## 📊 Dataset

The dataset consists of **59,983 news articles** gathered via `snscrape`:

* **Real News (~31k):** Sourced from Anadolu Agency (AA), Habertürk, TRT Haber, BBC Türkçe, etc.
* **Fake/Satirical News (~28k):** Sourced from Zaytung, Kramponnet, Kaparoz, Resmi Gaste, etc.

*Note: Due to strict disinformation laws in Turkey, the "fake news" category heavily relies on satirical news outlets to simulate deceptive content.*

## 🛠 Tech Stack & Tools

* **Language:** Python
* **NLP Processing:** [Zemberek](https://github.com/ahmetaa/zemberek-nlp) (Turkish NLP Library for lemmatization and normalization)
* **Data Gathering:** Snscrape (Twitter/X scraping)
* **Visualization:** Graphviz, Matplotlib
* **Machine Learning:** Scikit-learn (Logistic Regression, RF, XGBoost)
* **Deep Learning:** TensorFlow/Keras (CNN, LSTM)

## 🚀 Methodology

1.  **Data Collection:** Scraped ~60k tweets/articles from identified authentic and satirical sources.
2.  **Preprocessing:** * Removal of hyperlinks, punctuation, and emojis.
    * Stop-word filtering (Turkish-stop-words).
    * **Lemmatization** using the Zemberek library to handle Turkish agglutination.
3.  **Feature Extraction:** TF-IDF and Word Embeddings.
4.  **Modeling:** Training and tuning 5 distinct models.
5.  **Evaluation:** Comparing Accuracy, F1-Score, and AUC.

## 📈 Results

| Model | Accuracy (Test) | F1 Score | AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **91.40%** | **0.91** | **0.97** |
| XGBoost | 89.67% | 0.89 | 0.96 |
| CNN (Custom) | 88.17% | - | 0.96 |
| Random Forest | 82.00% | 0.85 | 0.92 |
| LSTM | *Subpar Performance* | - | - |

* **Key Insight:** Logistic Regression proved to be the most effective and efficient model for this specific binary text classification task.
* **Deep Learning Note:** While the CNN model yielded competitive results (~88%), the LSTM model struggled with the dataset, likely due to overfitting or data compatibility issues.

## 💻 Usage

To reproduce the results or train the models on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/username/turkish-fake-news-detection.git](https://github.com/username/turkish-fake-news-detection.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the preprocessing script:**
    ```bash
    python src/preprocessing.py
    ```

4.  **Train the models:**
    ```bash
    python src/train_models.py
    ```

## 🔮 Future Work

* incorporating multi-modal data (images + text).
* Expanding the dataset with more diverse "fake" news sources beyond satire.
* Testing Transformer-based models (e.g., BERTurk) for potentially higher accuracy.

## 📜 Reference

If you use this work, please cite our paper:
> *Sinal, S. E., Memişoğlu, A. K., & Üzüm, E. (2025). The Identification of Satirical Fake News in Turkey. Istanbul Kultur University.*
