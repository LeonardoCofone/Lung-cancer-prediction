# LUNG CANCER PREDICTION

**Powered by Leonardo Cofone**

This repository presents an end-to-end machine learning pipeline for predicting lung cancer using clinical and behavioral data. Leveraging a powerful ensemble of optimized models, we deliver a highly accurate classifier with state-of-the-art performance. The project features full preprocessing, robust model selection, threshold tuning, and balanced training.

---

## Project Highlights

* **Dataset**: Real-world survey data see the project on [Kaggle](https://www.kaggle.com/code/zlatan599/f1-0-98-lung-cancer-prediction).
* **Preprocessing**: Encoding, duplicate removal, standardization, resampling (RandomOverSampler).
* **Model Selection**: Evaluation of 6 models:

  * Logistic Regression
  * Random Forest ⭐
  * Support Vector Machine (SVM)
  * Gradient Boosting ⭐
  * Decision Tree
  * XGBoost
* **Hyperparameter Tuning**: Extensive `RandomizedSearchCV` across all models.
* **Ensemble Methods**:

  * **VotingClassifier**
  * **StackingClassifier**
* **Threshold Optimization**: ROC analysis + Youden's J statistic.
* **Performance**:

  * F1 Score: **0.98**
  * AUC: **0.996**

---

**Features**:

* Demographics: `AGE`, `GENDER`
* Risk factors: `SMOKING`, `ALCOHOL CONSUMING`, `PEER_PRESSURE`
* Symptoms: `WHEEZING`, `COUGHING`, `CHEST PAIN`, etc.

---

## 🌐 Data Preprocessing

```python
# Drop duplicates
...
# Encode categorical variables
...
# Normalize 'AGE'
...
# Oversample using RandomOverSampler
...
```

Train/test split:

```python
X_train: (357, 15)
X_test:  (119, 15)
```

---

## 🔧 Model Training & Optimization

Each model was wrapped in a `Pipeline` and tuned via `RandomizedSearchCV`:

| Model               | F1 Score |
| ------------------- | -------- |
| Logistic Regression | 0.91     |
| Random Forest       | 0.94 ⭐   |
| SVM                 | 0.98     |
| Gradient Boosting   | 0.94 ⭐   |
| Decision Tree       | 0.93     |
| XGBoost             | 0.93     |

> ⭐ Best models selected: **Random Forest** & **Gradient Boosting**

---

## ⚖️ Ensemble Learning

### 🔹 VotingClassifier

Combines predictions via majority vote.

### 🔹 StackingClassifier

Learns optimal combination of base models with a logistic regression meta-model:

```python
base_learners = [
    ('svm', best_svm_clf),
    ('gb', best_gb_clf),
    ('xgb', best_xgb_clf)
]
stacking_clf = StackingClassifier(...)
```

---

## 🌐 Threshold Tuning

To maximize real-world effectiveness:

* Used **ROC Curve** analysis
* Computed **Youden's J Statistic** to find the optimal classification threshold

---

## 📊 Final Performance

**Test Set Evaluation**:

* **F1 Score**: 0.98
* **Precision/Recall Balance**: Excellent
* **AUC**: 0.996

This model demonstrates robust generalization and practical clinical viability.

---

## 🧡 Author

**Leonardo Cofone**
Machine Learning & Deep Learning Specialist

---

## 🔗 Connect to me

* [LinkedIn](https://www.linkedin.com/in/leonardo-cofone-914228361/)
* [Kaggle](https://www.kaggle.com/zlatan599)
