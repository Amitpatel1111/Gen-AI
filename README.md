# IMDB Movie Review Sentiment Analysis

## Project Overview
This project focuses on building and evaluating sentiment analysis models for IMDB movie reviews. The goal is to classify movie reviews as either 'positive' or 'negative' using various text featurization techniques and machine learning classifiers.

## Dataset
The dataset used is a subset of the IMDB Movie Reviews dataset, consisting of 10,000 movie reviews, each labeled with a 'positive' or 'negative' sentiment.

## Methodology

### 1. Data Preprocessing
Before training the models, the raw text reviews underwent several preprocessing steps:
*   **HTML Tag Removal**: HTML tags (e.g., `<br /><br />`) were removed from the review text.
*   **Lowercasing**: All text was converted to lowercase to ensure consistency.
*   **Stopword Removal**: Common English stopwords (e.g., 'the', 'is', 'a') were removed to focus on more significant words.

### 2. Feature Extraction
Different techniques were employed to convert the text data into numerical features suitable for machine learning models:
*   **Bag of Words (BoW)**: Text was converted into a matrix of token counts using `CountVectorizer`.
    *   Initially, with default settings.
    *   Then, with `max_features=3000`.
    *   **N-grams**: BoW was also applied with `ngram_range=(1,2)` and `max_features=5000` to capture word sequences.
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**: Text was transformed into TF-IDF representations using `TfidfVectorizer`, which weights words based on their importance in the document and corpus.
*   **Word2Vec Embeddings**: A `Word2Vec` model was trained on the preprocessed training reviews to generate dense vector representations of words. Review embeddings were then created by averaging the vectors of the words within each review.

### 3. Classification Models
The following classifiers were used:
*   **Gaussian Naive Bayes**: Applied to Bag of Words features.
*   **Random Forest Classifier**: Applied to Bag of Words, N-gram, TF-IDF, and Word2Vec features.

## Results and Performance

| Feature Extraction Method | Classifier            | Accuracy Score |
| :------------------------ | :-------------------- | :------------- |
| Bag of Words              | Gaussian Naive Bayes  | 0.632          |
| Bag of Words              | Random Forest         | 0.846          |
| Bag of Words (max_features=3000) | Random Forest         | 0.835          |
| N-grams (ngram_range=(1,2), max_features=5000) | Random Forest         | 0.844          |
| TF-IDF                    | Random Forest         | 0.842          |
| Word2Vec Embeddings       | Random Forest         | 0.767          |

The Random Forest Classifier consistently performed well across most featurization techniques, with the standard Bag of Words yielding the highest accuracy in this comparison. Word2Vec embeddings, while powerful, showed a slightly lower accuracy in this specific setup compared to count-based methods.

## Visualizations
The project includes visualizations to better understand the data and model performance:
*   **Sentiment Distribution**: A bar chart showing the count of positive and negative reviews in the dataset.
*   **Confusion Matrix**: Heatmaps illustrating the true positives, true negatives, false positives, and false negatives for both the Gaussian Naive Bayes and Word2Vec-Random Forest models.
*   **Real vs. Predicted Sentiment Distribution**: Grouped bar charts comparing the actual sentiment distribution with the predicted sentiment distribution for each model, highlighting any class imbalance in predictions.

## How to Use (Google Colab)
1.  **Open the Notebook**: Upload and open the `.ipynb` file in Google Colab.
2.  **Run Cells**: Execute the cells sequentially. Make sure all necessary libraries (e.g., `pandas`, `numpy`, `scikit-learn`, `nltk`, `seaborn`, `gensim`) are installed and imported. The notebook will guide you through the data loading, preprocessing, model training, and evaluation steps.
3.  **Interact**: Feel free to modify parameters or explore different models to enhance performance.

## Libraries Used
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `nltk`
*   `scikit-learn`
*   `gensim`
