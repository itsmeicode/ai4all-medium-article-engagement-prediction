# AI4ALL Project

## Predicting Medium Article Engagement

Use machine learning to classify Medium articles by engagement level and predict the number of claps using text features, author history, and article metadata.

---

### About the Project

This project explores both classification and regression techniques to predict engagement on Medium articles, where engagement is measured using **claps**. Predictions leverage a combination of text-based features (title and subtitle), author reputation, and article metadata (tags, reading time, responses, publication date, etc.). The work encompasses data preprocessing, feature engineering, model training, evaluation, and interpretation.

The motivation comes from challenges faced by modern media organizations adapting to AI-driven content discovery. As these companies evolve, the central question is: how can we better understand and anticipate what makes content successful? Using historical engagement data, the project aims to uncover patterns and features that drive high engagement, providing actionable insights for content strategy and editorial decision-making.

---

### Models Used

#### Classification (High vs. Low Engagement)
- Random Forest Classifier  
- Balanced class weights to address label imbalance  
- Evaluated with accuracy, precision, recall, F1-score, and ROC AUC  

#### Regression (Predicting Clap Counts)
- Gradient Boosting Regressor  
- Predicts log-transformed clap counts  
- Evaluated with MAE, RMSE, and R²  

---

### Tools & Libraries

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- NLTK (VADER sentiment analysis)  
- SciPy  

---

### Dataset

- **Source:** [Medium Data Science Articles (2021 dataset)](https://www.kaggle.com/datasets/viniciuslambert/medium-2021-data-science-articles-dataset/data)  
- **Raw Data:** 47,660 articles  
- **Final Modeling Dataset:** 13,728 articles with complete text and metadata  

**Included Fields:**  
Title, subtitle, author, author page, claps, responses, reading time, tags, publication date, and URL.

---

### Key Features

- Combined title + subtitle text field  
- TF-IDF features (200 for classification; 100 each for title & subtitle in regression)  
- VADER sentiment scores  
- Text length features  
- Author average claps (captures author reputation)  
- Reading time and response count  
- One-hot encoded tags  
- Temporal features (month, weekday, weekend flag)  
- Log-transformed clap target for regression  

---

### Results

#### Classification (Random Forest Classifier)
- **Accuracy:** 0.91  
- **ROC AUC:** 0.875  
- **Precision (High Engagement):** 0.81  
- **Recall (High Engagement):** 0.81  

**Key Predictors:**  
Author average claps, text length, reading time, sentiment, and temporal features.

#### Regression (Gradient Boosting Regressor)
- **MAE:** 1.25  
- **RMSE:** 1.55  
- **R²:** 0.32  

**Key Predictors:**  
Response count, reading time, subtitle length, title TF-IDF components, and tag category.

---

### Insights

- Author reputation is the strongest indicator of high engagement.  
- Longer and more detailed articles receive more claps.  
- Sentiment plays a moderate role.  
- Timing matters, but less than content and author features.  
- Classification provides highly reliable high/low engagement predictions, while regression captures overall patterns in clap counts.

---

### Next Steps

- Incorporate topic modeling and transformer-based embeddings  
- Experiment with XGBoost, LightGBM, and deep learning models  
- Improve class imbalance handling (SMOTE, focal loss)  
- Add model explainability with SHAP values  
- Build a prediction API for editorial tools  
